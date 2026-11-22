"""Scene tracer: camera, group BVH acceleration, shading, shadows, reflections, post FX."""
import numpy as np
from .render_core import P_sphere, P_ellipsoid, P_box, P_cyl, P_capsule, P_cone, P_plane, Prim, EPS, INF, euler_to_mat, trs, quat_to_mat

# ------------------------------------------------------------------ camera
def look_at(eye, target, up=(0,1,0)):
    eye = np.array(eye, np.float64); target = np.array(target, np.float64)
    fwd = target - eye
    fwd = fwd / (np.linalg.norm(fwd)+1e-12)
    up = np.array(up, np.float64)
    right = np.cross(fwd, up); 
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(fwd, np.array([0,0,1.0]))
    right = right / (np.linalg.norm(right)+1e-12)
    upv = np.cross(right, fwd)
    return eye, fwd, right, upv


class Camera:
    def __init__(self, eye, target, fov_y=42.0, roll=0.0, dof=None):
        self.eye, self.fwd, self.right, self.upv = look_at(eye, target)
        self.roll = roll
        if roll != 0:
            c, s = np.cos(roll), np.sin(roll)
            r2 = self.right*np.cos(c)[..., None] if np.isscalar(c) else self.right*c + self.upv*s
            u2 = self.upv*c - self.right*s
            self.right, self.upv = r2, u2
        self.fov_y = fov_y
        self.dof = dof  # (focus_dist, fstop_scale) or None

    def gen_rays(self, W, H, jitter=1.0, rng=None, dof_samples=1):
        ys, xs = np.mgrid[0:H, 0:W]
        px = (xs.ravel().astype(np.float64) + rng.random(W*H) - 0.5) if jitter and rng is not None else xs.ravel().astype(np.float64)
        py = (ys.ravel().astype(np.float64) + rng.random(W*H) - 0.5) if jitter and rng is not None else ys.ravel().astype(np.float64)
        # NDC -> camera space
        aspect = W / H
        tanf = np.tan(np.radians(self.fov_y) / 2)
        # vertical fov: screen y from +1 (top) to -1 (bottom)
        sy = 1.0 - 2.0 * (py + 0.5) / H
        sx = 2.0 * (px + 0.5) / W - 1.0
        sx *= aspect * tanf
        sy *= tanf
        d = (self.fwd[None] + sx[:, None]*self.right[None] + sy[:, None]*self.upv[None])
        d = d / (np.linalg.norm(d, axis=1, keepdims=True)+1e-12)
        o = np.broadcast_to(self.eye, d.shape).copy()
        if self.dof is not None:
            focus, strength = self.dof
            # jitter origin on aperture plane
            ap = strength * 0.05
            ro = rng.random((W*H, 2))*2 - 1
            r = np.sqrt(ro[:, 0]**2 + ro[:, 1]**2)
            phi = np.arctan2(ro[:, 1], ro[:, 0])
            off = np.zeros_like(o)
            off += self.right[None] * (np.cos(phi)*r*ap)[:, None]
            off += self.upv[None]   * (np.sin(phi)*r*ap)[:, None]
            # move focal point along current ray, recompute dir from new origin
            t = focus / np.maximum((d * self.fwd[None]).sum(1), 0.05)
            fp = o + d * t[:, None]
            o = o + off
            d = fp - o
            d = d / (np.linalg.norm(d, axis=1, keepdims=True)+1e-12)
        return o, d


class Group:
    """Bounding group of prims with per-frame bounding sphere."""
    def __init__(self, prims):
        self.prims = prims

    def update_bounds(self):
        cs = np.array([p.bcenter for p in self.prims])
        rs = np.array([p.brad for p in self.prims])
        lo = (cs - rs[:, None]).min(0)
        hi = (cs + rs[:, None]).max(0)
        c = (lo + hi)/2
        self.c = c
        self.r = float(np.linalg.norm(hi - c)) + 1e-3


def intersect_groups(o, d, groups, best_t, occlusion=False, light_t=None, max_groups=None):
    """Trace against groups. If occlusion: return shadow mask bool.
    Else returns updated best_t, prim index per pixel (-1 none), and list hit flags."""
    N = o.shape[0]
    prim_idx = np.full(N, -1, np.int32)
    if occlusion:
        shadowed = np.zeros(N, bool)
    else:
        shadowed = None
    # per-group bounding sphere check
    gs = []
    for gi, g in enumerate(groups):
        oc = o - g.c[None]
        A = (d*d).sum(1)
        B = 2.0*(oc*d).sum(1)
        C = (oc*oc).sum(1) - g.r*g.r
        disc = B*B - 4*A*C
        m = (disc > 0) & (B < 0)   # ray moving towards sphere center
        t_close = -B / (2*A + 1e-12)
        m &= (t_close < best_t + g.r) if not occlusion else (t_close < (light_t if light_t is not None else 1e9))
        m &= (t_close > -g.r)
        if m.any():
            gs.append(gi)
    for gi in gs:
        g = groups[gi]
        oc_ref = o - g.c[None]
        B = 2.0*(oc_ref*d).sum(1)
        C = (oc_ref*oc_ref).sum(1) - g.r*g.r
        disc = B*B - 4*C
        near = (-B - np.sqrt(np.maximum(disc, 0)))*0.5
        far = (-B + np.sqrt(np.maximum(disc, 0)))*0.5
        if occlusion:
            m = (disc > 0) & (near < (light_t if light_t is not None else 1e9)) & (far > 0) & (~shadowed)
        else:
            m = (disc > 0) & (near < best_t) & (far > 0) & (~np.isinf(best_t) | True)
        if not m.any():
            continue
        sel = np.where(m)[0]
        o_s = o[sel]; d_s = d[sel]
        bt_s = best_t[sel]
        for pi, p in enumerate(g.prims):
            # local ray
            Miv = p.Minv
            R = Miv[:3, :3]
            T = Miv[:3, 3]
            o_l = o_s @ R.T + T
            d_l = d_s @ R.T
            # normalize? t mapped consistently in world units since R combines rotation+scale -> NOT unit.
            tl, hitm = intersect_unit2(p, o_l, d_l, bt_s)
            if not hitm.any():
                continue
            hit_sel = sel[hitm]
            if occlusion:
                shadowed[hit_sel] = True
                if shadowed.all():
                    return prim_idx, shadowed
                # shrink working set for speed
                keep = ~shadowed[sel]
                if keep.sum() != len(sel):
                    sel = sel[keep]; o_s = o[sel]; d_s = d[sel]; bt_s = best_t[sel]
                    continue
            else:
                # world-space t equals local t only if d not normalized in local; we kept unnormalized so
                # t_local along d_l corresponds to t_world along d (same scale factor because o_l+t d_l maps back).
                # verify: world = M(o_l + t d_l) = o + t d. YES. But bt_s comparison uses world t. consistent.
                upd = tl[hitm]
                better_mask = np.zeros(len(sel), bool)
                better_mask[hitm] = True
                # only where actually smaller than current best
                cur = best_t[sel]
                better = upd < cur
                new_mask = np.zeros(len(sel), bool)
                new_mask[hitm] = better
                new_sel2 = sel[new_mask]
                best_t[new_sel2] = upd[better]
                tw = tl[hitm][better]
                idx_n = hit_sel[better]
                # map global prim id
                prim_idx[idx_n] = pi_global(g, p)
    return prim_idx, shadowed


GLOBAL_PRIMS = []       # flat list; groups hold indices into this
def pi_global(g, p):
    # resolved at scene build time: group stores prim->global idx
    return g._pidx[g.prims.index(p)]


# faster path: merged into less copy-heavy helper
def intersect_unit2(p, o_l, d_l, best_t):
    from .render_core import EPS
    typ = p.type
    N = o_l.shape[0]
    t_out = np.full(N, INF, np.float64)
    hit = np.zeros(N, bool)
    bt = best_t
    if typ == 'sphere':
        b = (o_l*d_l).sum(1)
        c = (o_l*o_l).sum(1) - 1.0
        disc = b*b - c
        ok = disc > 0
        sq = np.sqrt(np.maximum(disc, 0.0))
        t = -b - sq
        m = ok & (t > EPS*10) & (t < bt)
        t_out[m] = t[m]; hit = m
    elif typ == 'box':
        inv = 1.0 / (d_l + np.where(np.abs(d_l) < 1e-9, 1e-9, 0.0))
        t0 = (-1.0 - o_l) * inv
        t1 = ( 1.0 - o_l) * inv
        tmin = np.minimum(t0, t1).max(1)
        tmax = np.maximum(t0, t1).min(1)
        m = (tmax > tmin) & (tmin > EPS*10) & (tmin < bt)
        t_out[m] = tmin[m]; hit = m
    elif typ == 'cyl':
        ox, oz, oy = o_l[:, 0], o_l[:, 2], o_l[:, 1]
        dx, dz, dy = d_l[:, 0], d_l[:, 2], d_l[:, 1]
        a = dx*dx + dz*dz
        b = 2.0*(ox*dx + oz*dz)
        c = ox*ox + oz*oz - 1.0
        disc = b*b - 4*a*c
        ok = (disc > 0) & (np.abs(a) > 1e-12)
        sq = np.sqrt(np.maximum(disc, 0.0))
        t1 = (-b - sq)/(2*a + 1e-12)
        t2 = (-b + sq)/(2*a + 1e-12)
        y1 = oy + t1*dy
        v1 = ok & (t1 > EPS*10) & (t1 < bt) & (np.abs(y1) <= 0.5)
        ty = ((0.5 - oy) * np.sign(dy + 1e-12) * (dy > 0) + (-0.5 - oy) * (dy < 0)) / (np.abs(dy) + 1e-12)
        xx = ox + ty*dx; zz = oz + ty*dz
        vcap = (np.abs(dy) > 1e-9) & (ty > EPS*10) & (ty < bt) & (xx*xx + zz*zz <= 1.0)
        best = np.where(v1, t1, INF); best = np.where(vcap, np.minimum(best, ty), best)
        mm = np.isfinite(best) & (best < INF)
        t_out[mm] = best[mm]; hit = mm
    elif typ == 'caps':
        hx = 0.6
        ox, oy, oz = o_l[:, 0], o_l[:, 1], o_l[:, 2]
        dx, dy, dz = d_l[:, 0], d_l[:, 1], d_l[:, 2]
        best = np.full(N, INF, np.float64)
        for cy in (-hx, hx):
            ocy = oy - cy
            b = 2.0*(ox*dx + ocy*dy + oz*dz)
            c = ox*ox + ocy*ocy + oz*oz - 1.0
            disc = b*b - 4*c
            ok = disc > 0
            sq = np.sqrt(np.maximum(disc, 0.0))
            t = -b - sq
            yend = oy + t*dy
            vk = ok & (t > EPS*10) & (t < bt) & ((yend <= cy) if cy == -hx else (yend >= cy))
            best = np.where(vk, np.minimum(best, t), best)
        a = dx*dx + dz*dz
        b = 2.0*(ox*dx + oz*dz)
        c = ox*ox + oz*oz - 1.0
        disc = b*b - 4*a*c
        ok = (disc > 0) & (np.abs(a) > 1e-12)
        sq = np.sqrt(np.maximum(disc, 0.0))
        t = (-b - sq)/(2*a + 1e-12)
        yend = oy + t*dy
        vk = ok & (t > EPS*10) & (t < bt) & (np.abs(yend) <= hx)
        best = np.where(vk, np.minimum(best, t), best)
        mm = np.isfinite(best) & (best < INF)
        t_out[mm] = best[mm]; hit = mm
    elif typ == 'cone':
        oy, dy = o_l[:, 1], d_l[:, 1]
        ox, oz, dx, dz = o_l[:, 0], o_l[:, 2], d_l[:, 0], d_l[:, 2]
        A = dx*dx + dz*dz - dy*dy
        B = 2*(ox*dx + oz*dz) + 2*dy*(0.5 - oy)
        C = ox*ox + oz*oz - (0.5 - oy)**2
        disc = B*B - 4*A*C
        ok = disc > 0
        sq = np.sqrt(np.maximum(disc, 0.0))
        best = np.full(N, INF, np.float64)
        for t in ((-B - sq)/(2*A + 1e-12), (-B + sq)/(2*A + 1e-12)):
            yend = oy + t*dy
            vk = ok & (t > EPS*10) & (t < bt) & (yend >= -0.5) & (yend <= 0.5)
            best = np.where(vk, np.minimum(best, t), best)
        tb = (-0.5 - oy)/(dy + 1e-12)
        x2 = ox + tb*dx; z2 = oz + tb*dz
        vkb = ok & (tb > EPS*10) & (tb < bt) & (x2*x2 + z2*z2 <= 1.0)
        best = np.where(vkb, np.minimum(best, tb), best)
        mm = np.isfinite(best) & (best < INF)
        t_out[mm] = best[mm]; hit = mm
    elif typ == 'plane':
        n = p['n']; pp = p['p']
        den = (d_l @ n)
        den = np.where(np.abs(den) < 1e-9, 1e-9, den)
        t = ((pp - o_l) @ n) / den
        m = (t > EPS*10) & (t < bt)
        t_out[m] = t[m]; hit = m
    return t_out, hit
