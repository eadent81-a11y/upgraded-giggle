"""GLIMMER STATION — custom cinematic renderer core.
Analytic raytracer: transformed primitives, group-BVH masking, shadows,
reflections, sky ambience, fog, bloom, filmic grade. float32, vectorized.
"""
import numpy as np

EPS = 1e-4
INF = 1e9


# ---------------------------------------------------------------- transforms
def quat_to_mat(q):
    """q = (x,y,z,w) rotation quaternion -> 3x3 (approx orthonormalized)."""
    x, y, z, w = q
    n = np.sqrt(x*x + y*y + z*z + w*w) + 1e-12
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], np.float64)


def euler_to_mat(rx, ry, rz, order='xyz'):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], np.float64)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], np.float64)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], np.float64)
    R = {'x':Rx, 'y':Ry, 'z':Rz}
    M = np.eye(3)
    for ax in order:
        M = R[ax] @ M
    return M


def trs(pos, rot_mat=None, scale=None):
    """Build 4x4 world matrix from position, rotation matrix, scale."""
    M = np.eye(4)
    if scale is not None:
        M[:3, :3] = np.asarray(scale, np.float64) * (rot_mat if rot_mat is not None else np.eye(3))
    elif rot_mat is not None:
        M[:3, :3] = rot_mat
    if pos is not None:
        M[:3, 3] = pos
    return M


def trs_inv(M):
    Ri = M[:3, :3].T          # rotation column-orthonormal (pure rotation * scale)
    s = np.linalg.norm(Ri, axis=0)              # per-axis scale
    Ro = Ri / np.where(np.abs(s) < 1e-8, 1e-8, np.abs(s))[None, :]
    t = -Ro @ (M[:3, 3] / np.where(np.abs(s) < 1e-8, 1e-8, s))
    Mi = np.eye(4)
    Mi[:3, :3] = Ro / np.where(np.abs(s) < 1e-8, 1.0, s)[None, :]
    Mi[:3, 3] = t
    return Mi


def mat_mul(A, B):
    return A @ B


# ---------------------------------------------------------------- primitives
# Each prim dict:
#  type: 'sphere'|'box'|'cyl'|'cone'|'caps'|'plane'
#  M (4x4), Minv (4x4)
#  mat: {'albedo':rgb, 'rough':float, 'refl':float, 'emis':rgb|None, 'spec':k}

def make_sphere(center, radius, mat):
    return {'type': 'sphere', 'M': trs(center, None, [radius]*3), 'mat': mat}


class Prim:
    __slots__ = ('type', 'M', 'Minv', 'mat', 'bcenter', 'brad')
    def __init__(self, type, M, mat, bcenter, brad):
        self.type = type; self.M = M; self.Minv = trs_inv(M); self.mat = mat
        self.bcenter = bcenter; self.brad = brad


def P_sphere(center, radius, mat):
    return Prim('sphere', trs(center, None, [radius]*3), mat, np.array(center, float), radius*1.35)

def P_ellipsoid(center, rot, scale, mat):
    Md = trs(center, rot, scale)
    rad = float(np.linalg.norm(scale))
    return Prim('sphere', Md, mat, np.array(center, float), rad*1.35)

def P_box(center, rot, size, mat):
    Md = trs(center, rot, size)
    rad = float(np.linalg.norm(size))
    return Prim('box', Md, mat, np.array(center, float), rad*1.1)

def P_cyl(center, rot, radius, height, mat):
    """unit cylinder radius1 height1 centered at origin (y in [-.5,.5]).
    height = total height -> scale y=height."""
    Md = trs(center, rot, [radius, height, radius])
    rad = float(np.sqrt(radius*radius + (height*0.5)**2))
    return Prim('cyl', Md, mat, np.array(center, float), rad*1.15)

def P_capsule(center, rot, radius, length, mat):
    """capsule along local Y, total length=length (2*half), cap centers at +-length/2."""
    Md = trs(center, rot, [radius, length*0.5 + radius, radius])
    rad = length*0.5 + 2*radius
    return Prim('caps', Md, mat, np.array(center, float), rad*1.15)

def P_cone(center, rot, radius, height, mat):
    """cone base radius r, apex at +h/2 in local Y, base at -h/2."""
    Md = trs(center, rot, [radius, height, radius])
    rad = float(np.sqrt(radius*radius + (height*0.5)**2))
    return Prim('cone', Md, mat, np.array(center, float), rad*1.15)

def P_plane(point, normal, mat, size=1e4):
    """Infinite plane; store as prim with custom transform: local z = normal dir.
    We implement plane directly with point+n; bounding = huge."""
    p = {'type': 'plane', 'M': None, 'Minv': None, 'mat': mat,
         'p': np.array(point, np.float64), 'n': np.array(normal, np.float64)/np.linalg.norm(normal),
         'bcenter': np.array(point, np.float64), 'brad': 1e9}
    return p


# ---------------------------------------------------------------- ray/prim intersect (unit space)
def isect_unit(prim, o, d, best_t):
    """Vectorized intersection of rays (already in unit local space) with prim.type.
    Returns (t, hit_mask) where t valid only where hit."""
    tbest = np.where(best_t < INF, best_t, INF)
    typ = prim.type
    N = o.shape[0]
    t_out = np.full(N, INF, np.float64)
    hit_any = np.zeros(N, bool)
    if typ == 'sphere':
        a = 1.0  # unit sphere after non-uniform? we bake scale into M, so unit space exact sphere
        b = 2.0 * (o*d).sum(1)
        c = (o*o).sum(1) - 1.0
        disc = b*b - 4.0*c
        ok = disc > 0
        sq = np.sqrt(np.maximum(disc, 0.0))
        t = (-b - sq) * 0.5
        valid = ok & (t > EPS) & (t < tbest)
        t_out[valid] = t[valid]; hit_any = valid
    elif typ == 'box':
        inv = 1.0 / (d + np.where(np.abs(d) < 1e-8, 1e-8, 0.0))
        t0 = (-1.0 - o) * inv
        t1 = ( 1.0 - o) * inv
        tmin = np.minimum(t0, t1).max(1)
        tmax = np.maximum(t0, t1).min(1)
        valid = (tmax > tmin) & (tmax > EPS) & (tmin < tbest)
        # entering (tmin>0) or inside box (tmin<0): use tmin anyway
        tf = tmin
        inside = (tf < EPS)
        tf2 = np.where(inside & (tmax > EPS), tmax*1.0, tf)
        valid = valid & ((tf > EPS) | inside)
        t_out[valid] = tf2[valid]; hit_any = valid
    elif typ == 'cyl':
        ox, oz = o[:, 0], o[:, 2]
        dx, dz = d[:, 0], d[:, 2]
        a = dx*dx + dz*dz
        b = 2.0*(ox*dx + oz*dz)
        c = ox*ox + oz*oz - 1.0
        disc = b*b - 4*a*c
        ok = (disc > 0) & (np.abs(a) > 1e-8)
        sq = np.sqrt(np.maximum(disc, 0.0))
        t1 = (-b - sq)/(2*a + 1e-12)
        t2 = (-b + sq)/(2*a + 1e-12)
        y1 = o[:, 1] + t1*d[:, 1]
        valid1 = ok & (t1 > EPS) & (t1 < tbest) & (np.abs(y1) <= 0.5)
        # caps (local y = +-0.5)
        ty1 = ( 0.5 - o[:, 1]) / (d[:, 1] + 1e-12)
        ty0 = (-0.5 - o[:, 1]) / (d[:, 1] + 1e-12)
        x1 = ox + ty1*dx; z1 = oz + ty1*dz; r1 = x1*x1 + z1*z1
        x0 = ox + ty0*dx; z0 = oz + ty0*dz; r0 = x0*x0 + z0*z0
        vcap1 = (ty1 > EPS) & (ty1 < tbest) & (r1 <= 1.0)
        vcap0 = (ty0 > EPS) & (ty0 < tbest) & (r0 <= 1.0)
        best = t_out.copy()
        best = np.where(valid1, np.minimum(best, t1), best)
        best = np.where(vcap1, np.minimum(best, ty1), best)
        best = np.where(vcap0, np.minimum(best, ty0), best)
        hit_any = (best < t_out) | (valid1 & (np.isfinite(t1)))
        t_out = best
        hit_any = (t_out < INF) & (np.isfinite(t_out))
        valid1 = valid1 | vcap1 | vcap0
        hit_any = valid1 & (t_out > EPS) & (t_out < INF)
    elif typ == 'caps':
        # capsule: unit space = segment along y from -h to h with radius 1-something? We bake as scaled unit sphere with caps implied.
        # Approximate capsule as: intersection with scaled sphere (body) — good enough visually for short capsules,
        # but do proper: distance to y-axis clamp.
        hx = 0.6   # assumed half-length fraction of unit-space; visual approximation ok since we scale by (r, len/2+r, r)
        # proper: closest point on segment y in [-hx,hx]
        dy = d[:, 1]
        # solve |o + t d - clamp_y|^2 ; use quadratic withclosest approach
        # distance^2 = |oxy|^2 + (oy + t dy - cy)^2 where cy = clamp projection
        # Use standard: a = dx^2+dz^2 + dy^2 (full), treat as two spheroid caps via 2 spheres at +-hx and cylinder between.
        # Simpler robust: try sphere at +hx and -hx and cylinder y in [-hx,hx]; take min.
        best = t_out
        for cy in (-hx, hx):
            oc = o.copy()
            oc[:, 1] = o[:, 1] - cy
            b = 2.0*(oc*d).sum(1)
            c = (oc*oc).sum(1) - 1.0
            disc = b*b - 4*c
            ok = disc > 0
            sq = np.sqrt(np.maximum(disc, 0.0))
            t = (-b - sq)*0.5
            yend = o[:, 1] + t*dy
            vk = ok & (t > EPS) & (t < tbest)
            tmask = np.where(cy == -hx, yend <= -hx, yend >= hx)
            vk = vk & tmask
            best = np.where(vk, np.minimum(best, t), best)
        # cylinder y in [-hx, hx]
        ox, oz = o[:, 0], o[:, 2]
        dx, dz = d[:, 0], d[:, 2]
        a = dx*dx + dz*dz
        b = 2.0*(ox*dx + oz*dz)
        c = ox*ox + oz*oz - 1.0
        disc = b*b - 4*a*c
        ok = (disc > 0) & (np.abs(a) > 1e-8)
        sq = np.sqrt(np.maximum(disc, 0.0))
        t = (-b - sq)/(2*a + 1e-12)
        yend = o[:, 1] + t*dy
        vk = ok & (t > EPS) & (t < tbest) & (np.abs(yend) <= hx)
        best = np.where(vk, np.minimum(best, t), best)
        # end caps discs
        for cy in (-hx, hx):
            tt = (cy - o[:, 1]) / (dy + 1e-12)
            x2 = ox + tt*dx; z2 = oz + tt*dz
            vk = (tt > EPS) & (tt < tbest) & (x2*x2 + z2*z2 <= 1.0)
            vk = vk & ((cy == hx) if False else np.full(N, True))
            # only for slabs facing correct dir:
            vk = vk & ((dy > 0) if cy == -hx else (dy < 0) if cy == hx else np.full(N, True))
            best = np.where(vk, np.minimum(best, tt), best)
        t_out = best
        hit_any = (t_out < INF) & np.isfinite(t_out)
    elif typ == 'cone':
        # unit cone: base circle r=1 at y=-0.5, apex y=+0.5
        oy, dy = o[:, 1], d[:, 1]
        ox, oz = o[:, 0], o[:, 2]
        dx, dz = d[:, 0], d[:, 2]
        # from y -0.5..0.5, radius k(y) = (0.5 - y)  (r=1 at y=-.5)
        k = 1.0
        # solve: (ox+t dx)^2 + (oz+t dz)^2 = (0.5 - y)^2 where y = oy + t dy
        A = dx*dx + dz*dz - dy*dy
        B = 2*(ox*dx + oz*dz) + 2*oy*dy + dy  # dy term: -2*(0.5 - oy - t dy)*dy -> +2*dy*oy + t*2dy^2... recompute below
        # radius(y) = (0.5 - y) so (0.5 - y)^2 = (0.5 - oy - t dy)^2
        # => ox^2+... = 0.25 - oy - t dy + oy^2 + 2 t oy dy + t^2 dy^2 ... careful
        # f(t)= (ox+t dx)^2+(oz+t dz)^2 - (0.5-oy-t dy)^2 = 0
        # A = dx^2+dz^2 - dy^2
        A = dx*dx + dz*dz - dy*dy
        B = 2*(ox*dx + oz*dz) + 2*dy*(0.5 - oy)
        C = ox*ox + oz*oz - (0.5 - oy)**2
        disc = B*B - 4*A*C
        ok = disc > 0
        sq = np.sqrt(np.maximum(disc, 0.0))
        ra = (-B - sq)/(2*A + 1e-12)
        rb = (-B + sq)/(2*A + 1e-12)
        cand = []
        for t in (ra, rb):
            yend = oy + t*dy
            r_at = 0.5 - yend
            r_hit = np.sqrt((ox + t*dx)**2 + (oz + t*dz)**2)
            vk = ok & (t > EPS) & (t < tbest) & (yend >= -0.5) & (yend <= 0.5)
            cand.append((vk, t))
        # base cap
        tb = (-0.5 - oy)/(dy + 1e-12)
        x2 = ox + tb*dx; z2 = oz + tb*dz
        vkb = ok & (tb > EPS) & (tb < tbest) & (x2*x2 + z2*z2 <= 1.0)
        bcand = (vkb, tb)
        best = t_out
        for vk, t in cand + [bcand]:
            best = np.where(vk, np.minimum(best, t), best)
        t_out = best
        hit_any = (t_out < INF) & np.isfinite(t_out)
    elif typ == 'plane':
        n = prim['n']; p = prim['p']
        denom = o @ n if False else d @ n
        denom = np.asarray(denom, np.float64)
        den = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        t = ((p - o) @ n) / den
        valid = (t > EPS) & (t < tbest)
        t_out[valid] = t[valid]; hit_any = valid
    return t_out, hit_any


# ---------------------------------------------------- local-space normals
def unit_normal(prim, o_local, d_local, t):
    """returns world normal at hitpoint given local origin/dir and local t."""
    typ = prim.type
    Nn = o_local + t[:, None]*d_local
    if typ == 'sphere':
        n = Nn / (np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12)
    elif typ == 'box':
        ax = np.abs(Nn)
        comp = np.argmax(ax, axis=1)
        n = np.zeros_like(Nn)
        idx = np.arange(Nn.shape[0])
        n[idx, comp] = np.sign(Nn[idx, comp])
    elif typ == 'cyl':
        n = Nn.copy(); n[:, 1] = 0
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        side = ln[:, 0] > 1e-6
        n_side = n / (ln + 1e-12)
        n_top = np.zeros_like(n); n_top[:, 1] = 1.0
        y = Nn[:, 1]
        n = np.where((np.abs(y) > 0.4999)[:, None], n_top, n_side)
    elif typ == 'caps':
        hx = 0.6
        n = Nn.copy()
        # if |y|>hx: sphere cap normal from sphere center
        cap_hi = Nn[:, 1] > hx
        cap_lo = Nn[:, 1] < -hx
        n_mid = n.copy(); n_mid[:, 1] = 0
        lm = np.linalg.norm(n_mid, axis=1, keepdims=True) + 1e-12
        nsm = n_mid / lm
        n_hi = Nn - np.array([0, hx, 0])
        n_hi = n_hi / (np.linalg.norm(n_hi, axis=1, keepdims=True) + 1e-12)
        n_lo = Nn - np.array([0, -hx, 0])
        n_lo = n_lo / (np.linalg.norm(n_lo, axis=1, keepdims=True) + 1e-12)
        n = np.where(cap_hi[:, None], n_hi, np.where(cap_lo[:, None], n_lo, nsm))
    elif typ == 'cone':
        # side normal: perpendicular to slant
        # point p=(x,y,z): radial (x, z) direction + slope dy/dr = 1 (r = 0.5-y). n = normalize(x, dy/dr-comp, z)
        # slant slope: dr/dy = -1 => normal y-comp = +1 relative to radial: n=(x, 1, z)/|..| (surface cone)
        radial = np.stack([Nn[:, 0], Nn[:, 2]], 1)
        lr = np.linalg.norm(radial, axis=1) + 1e-12
        y_apex = Nn[:, 1] > 0.4999
        n_side = np.stack([Nn[:, 0]/lr, np.ones(Nn.shape[0])*1.0, Nn[:, 2]/lr], 1)
        n_side = n_side / (np.linalg.norm(n_side, axis=1, keepdims=True) + 1e-12)
        n_base = np.zeros_like(n_side); n_base[:, 1] = -1.0
        n = np.where(y_apex[:, None], np.array([0,1.0,0]), n_side)
        # for base cap hits (y ~ -0.5 and radial<1) use base normal
        n = np.where((np.abs(Nn[:, 1] + 0.5) < 1e-4)[:, None], n_base, n)
    else:
        n = np.zeros_like(Nn)
    Ml = prim.M[:3, :3]
    n_world = n @ Ml.T
    return n_world / (np.linalg.norm(n_world, axis=1, keepdims=True) + 1e-12)
