"""GLIMMER STATION renderer: scene model, raytracer, shading, post-processing."""
import numpy as np

EPS = 1e-4
INF = 1e9


# ================================================================ transforms
class Node:
    """Hierarchical transform node; world = parent.world @ TRS(local)."""
    __slots__ = ('pos', 'rot', 'scl', 'parent', 'world', 'anim', 'name')
    _next = [0]
    def __init__(self, pos=(0,0,0), rot=(0,0,0), scl=(1,1,1), parent=None, name=None):
        self.pos = np.array(pos, np.float64)
        self.rot = np.array(rot, np.float64)
        self.scl = np.array(scl, np.float64)
        self.parent = parent
        self.name = name or f"n{Node._next[0]}"; Node._next[0] += 1
        self.anim = {}
        self.world = np.eye(4)

    def update(self):
        for k, fn in self.anim.items():
            v = fn()
            if k == 'pos': self.pos = np.asarray(v, np.float64)
            elif k == 'rot': self.rot = np.asarray(v, np.float64)
            elif k == 'scl': self.scl = np.asarray(v, np.float64)
        W = trs4(self.pos, self.rot, self.scl)
        self.world = (self.parent.world @ W) if self.parent is not None else W
        return self.world


def trs4(pos, rot, scl):
    rx, ry, rz = rot
    cx, sx = np.cos(rx), np.sin(rx); cy, sy = np.cos(ry), np.sin(ry); cz, sz = np.cos(rz), np.sin(rz)
    R = np.array([
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy,   cy*sx,            cy*cx]], np.float64)
    M = np.eye(4)
    M[:3, :3] = R * np.asarray(scl)[None, :]
    M[:3, 3] = pos
    return M


def _inv4(M):
    Ri = M[:3, :3]
    s = np.linalg.norm(Ri, axis=0)
    s_safe = np.where(np.abs(s) < 1e-9, 1e-9, s)
    Rs = Ri / s_safe[None, :]
    Mi = np.eye(4)
    Mi[:3, :3] = Rs.T / s_safe[None, :]
    Mi[:3, 3] = -(Rs.T @ M[:3, 3]) / s_safe[None, :]
    return Mi


# ================================================================ materials / prims
def mat(albedo, rough=0.5, refl=0.0, emis=None, spec=0.5, sspec=0.04, nocast=False):
    return {'albedo': np.array(albedo, np.float64), 'rough': rough, 'refl': refl,
            'emis': None if emis is None else np.array(emis, np.float64), 'spec': spec, 'sspec': sspec,
            'nocast': nocast}


class Prim:
    __slots__ = ('type', 'M', 'Minv', 'mat', 'bc', 'br', 'node', 'local')
    def __init__(self, type_, local, matp, node=None):
        self.type = type_
        self.local = local
        self.mat = matp
        self.node = node
        self.M = (node.world @ local) if node is not None else local.copy()
        self.Minv = _inv4(self.M)
        c = self.M[:3, 3]
        ext = self.M[:3, :3].sum(1)
        self.bc = c.copy()
        self.br = max(float(np.linalg.norm(ext)), 0.01)

    def refresh(self):
        if self.node is None:
            return
        self.M = self.node.world @ self.local
        self.Minv = _inv4(self.M)
        c = self.M[:3, 3]
        ext = self.M[:3, :3].sum(1)
        self.bc = c.copy()
        self.br = max(float(np.linalg.norm(ext)), 0.01)


def pr(node, type_, matp, local=None):
    if local is None:
        local = np.eye(4)
    return Prim(type_, np.asarray(local, np.float64), matp, node)


def world_prim(type_, pos, rot, scl, matp):
    return Prim(type_, trs4(pos, rot, scl), matp, None)


# ================================================================ scene
class Scene:
    def __init__(self):
        self.nodes = []
        self.prim_nodes = []
        self.prim_static = []
        self.prim_groups = []
        self.lights = []
        self.sky_fn = None
        self.fog = None
        self.amb_top = np.array([0.34, 0.36, 0.44])
        self.amb_bot = np.array([0.10, 0.10, 0.13])
        self.root = Node(name='root')
        self.nodes.append(self.root)

    def node(self, name, pos=(0,0,0), rot=(0,0,0), scl=(1,1,1), parent=None):
        nd = Node(pos, rot, scl, parent if parent is not None else self.root, name)
        self.nodes.append(nd)
        return nd

    def attach(self, type_, node, local, matp, group=None):
        p = pr(node, type_, matp, local)
        self.prim_nodes.append(p)
        self._grp(group).append(p)
        return p

    def static(self, type_, pos, rot, scl, matp, group=None):
        p = world_prim(type_, pos, rot, scl, matp)
        self.prim_static.append(p)
        self._grp(group).append(p)
        return p

    def _grp(self, name):
        if name is None:
            name = '_default'
        for g in self.prim_groups:
            if g['name'] == name:
                return g['prims']
        self.prim_groups.append({'name': name, 'prims': []})
        return self.prim_groups[-1]['prims']

    def refresh(self):
        for nd in self.nodes:
            nd.update()
        for p in self.prim_nodes:
            p.refresh()


# ================================================================ intersection
def _local_ray(p, o, d):
    Ri = p.Minv[:3, :3]; Ti = p.Minv[:3, 3]
    return o @ Ri.T + Ti, d @ Ri.T


def _isect(p, o_l, d_l, best_t):
    N = o_l.shape[0]
    t_out = np.full(N, INF, np.float64)
    typ = p.type
    if typ == 'sphere':
        a = (d_l*d_l).sum(1)
        b = (o_l*d_l).sum(1) * 2.0
        c = (o_l*o_l).sum(1) - 1.0
        disc = b*b - 4*a*c
        ok = disc > 0
        sq = np.sqrt(np.maximum(disc, 0.0))
        t = (-b - sq) / (2*a + 1e-15)
        m = ok & (t > EPS) & (t < best_t)
        t_out[m] = t[m]
    elif typ == 'box':
        inv = 1.0 / (d_l + np.where(np.abs(d_l) < 1e-9, 1e-9, 0.0))
        t0 = (-1.0 - o_l) * inv
        t1 = ( 1.0 - o_l) * inv
        tmin = np.minimum(t0, t1).max(1)
        tmax = np.maximum(t0, t1).min(1)
        m = (tmax > tmin) & (tmin > EPS) & (tmin < best_t)
        t_out[m] = tmin[m]
    elif typ == 'cyl':
        ox, oy, oz = o_l[:, 0], o_l[:, 1], o_l[:, 2]
        dx, dy, dz = d_l[:, 0], d_l[:, 1], d_l[:, 2]
        a = dx*dx + dz*dz
        b = 2.0*(ox*dx + oz*dz)
        c = ox*ox + oz*oz - 1.0
        disc = b*b - 4*a*c
        ok = (disc > 0) & (np.abs(a) > 1e-12)
        sq = np.sqrt(np.maximum(disc, 0.0))
        t = (-b - sq) / (2*a + 1e-15)
        yend = oy + t*dy
        m = ok & (t > EPS) & (t < best_t) & (np.abs(yend) <= 0.5)
        alive = np.where(np.abs(dy) < 1e-9, 1e-9, dy)
        ty_top = (0.5 - oy) / alive
        ty_bot = (-0.5 - oy) / alive
        xx = ox + ty_top*dx; zz = oz + ty_top*dz
        mtop = (ty_top > EPS) & (ty_top < best_t) & (xx*xx + zz*zz <= 1.0) & (dy < 0)
        xx = ox + ty_bot*dx; zz = oz + ty_bot*dz
        mbot = (ty_bot > EPS) & (ty_bot < best_t) & (xx*xx + zz*zz <= 1.0) & (dy > 0)
        m2 = m | mtop | mbot
        comb = np.minimum(np.where(m, t, INF), np.minimum(np.where(mtop, ty_top, INF), np.where(mbot, ty_bot, INF)))
        t_out = np.where(m2 & (comb < best_t), comb, INF)
    elif typ == 'caps':
        hx = 0.6
        ox, oy, oz = o_l[:, 0], o_l[:, 1], o_l[:, 2]
        dx, dy, dz = d_l[:, 0], d_l[:, 1], d_l[:, 2]
        best = np.full(N, INF, np.float64)
        a = dx*dx + dz*dz
        b = 2.0*(ox*dx + oz*dz)
        c = ox*ox + oz*oz - 1.0
        disc = b*b - 4*a*c
        ok = (disc > 0) & (np.abs(a) > 1e-12)
        sq = np.sqrt(np.maximum(disc, 0.0))
        t = (-b - sq) / (2*a + 1e-15)
        yend = oy + t*dy
        m = ok & (t > EPS) & (t < best_t) & (np.abs(yend) <= hx)
        best = np.where(m, np.minimum(best, t), best)
        a2 = dx*dx + dy*dy + dz*dz
        for cy, side in ((-hx, -1.0), (hx, 1.0)):
            ocy = oy - cy
            b2 = 2.0*(ox*dx + ocy*dy + oz*dz)
            c2 = ox*ox + ocy*ocy + oz*oz - 1.0
            disc = b2*b2 - 4*a2*c2
            ok = disc > 0
            sq = np.sqrt(np.maximum(disc, 0.0))
            t = (-b2 - sq) / (2*a2 + 1e-15)
            yend = oy + t*dy
            m = ok & (t > EPS) & (t < best_t) & (side*yend >= side*cy)
            best = np.where(m, np.minimum(best, t), best)
        mm = best < INF
        t_out[mm] = best[mm]
    elif typ == 'cone':
        oy, dy = o_l[:, 1], d_l[:, 1]
        ox, oz, dx, dz = o_l[:, 0], o_l[:, 2], d_l[:, 0], d_l[:, 2]
        A = dx*dx + dz*dz - dy*dy
        B = 2*(ox*dx + oz*dz) + 2*dy*(0.5 - oy)
        C = ox*ox + oz*oz - (0.5 - oy)**2
        disc = B*B - 4*A*C
        ok = (disc > 0) & (np.abs(A) > 1e-12)
        sq = np.sqrt(np.maximum(disc, 0.0))
        best = np.full(N, INF, np.float64)
        for t in ((-B - sq)/(2*A + 1e-15), (-B + sq)/(2*A + 1e-15)):
            yend = oy + t*dy
            m = ok & (t > EPS) & (t < best_t) & (yend >= -0.5) & (yend <= 0.5)
            best = np.where(m, np.minimum(best, t), best)
        alive = np.where(np.abs(dy) < 1e-9, 1e-9, dy)
        tb = (-0.5 - oy)/alive
        x2 = ox + tb*dx; z2 = oz + tb*dz
        mb = (np.abs(dy) > 1e-9) & (tb > EPS) & (tb < best_t) & (x2*x2 + z2*z2 <= 1.0) & (dy > 0)
        best = np.where(mb, np.minimum(best, tb), best)
        mm = best < INF
        t_out[mm] = best[mm]
    return t_out


def _normal(p, o_l, d_l, t):
    hl = o_l + t[:, None]*d_l
    typ = p.type
    if typ == 'sphere':
        n = hl / (np.linalg.norm(hl, axis=1, keepdims=True) + 1e-15)
    elif typ == 'box':
        ax = np.abs(hl)
        comp = ax.argmax(1)
        n = np.zeros_like(hl)
        idx = np.arange(hl.shape[0])
        s = np.sign(hl[idx, comp]); s[s == 0] = 1.0
        n[idx, comp] = s
    elif typ == 'cyl':
        y = hl[:, 1]
        n = hl.copy(); n[:, 1] = 0
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        side_ok = ln[:, 0] > 1e-6
        n_s = n / (ln + 1e-15)
        n_t = np.zeros_like(hl); n_t[:, 1] = np.where(y >= 0, 1.0, -1.0)
        n = np.where((np.abs(y) > 0.49999)[:, None], n_t, n_s)
    elif typ == 'caps':
        hx = 0.6
        n = np.zeros_like(hl)
        n_mid = hl.copy(); n_mid[:, 1] = 0
        lm = np.linalg.norm(n_mid, axis=1, keepdims=True) + 1e-15
        nsm = n_mid / lm
        n_hi = hl - np.array([0, hx, 0]); n_hi /= (np.linalg.norm(n_hi, axis=1, keepdims=True)+1e-15)
        n_lo = hl - np.array([0,-hx, 0]); n_lo /= (np.linalg.norm(n_lo, axis=1, keepdims=True)+1e-15)
        n = np.where((hl[:, 1] > hx)[:, None], n_hi, np.where((hl[:, 1] < -hx)[:, None], n_lo, nsm))
    elif typ == 'cone':
        y_apex = hl[:, 1] > 0.4999
        radial = np.stack([hl[:, 0], hl[:, 2]], 1)
        lr = np.linalg.norm(radial, axis=1) + 1e-15
        n_side = np.stack([hl[:, 0]/lr, np.full(hl.shape[0], 1.0), hl[:, 2]/lr], 1)
        n_side /= (np.linalg.norm(n_side, axis=1, keepdims=True)+1e-15)
        n_base = np.zeros_like(n_side); n_base[:, 1] = -1.0
        n = np.where(y_apex[:, None], np.array([0,1.0,0.0]), n_side)
        n = np.where((np.abs(hl[:, 1] + 0.5) < 2e-4)[:, None], n_base, n)
    else:
        n = np.ones_like(hl)
    R = p.M[:3, :3]
    s = np.linalg.norm(R, axis=0)
    s_safe = np.where(np.abs(s) < 1e-9, 1e-9, s)
    Rs = R / s_safe[None, :]
    Nw = n @ Rs
    Nw /= (np.linalg.norm(Nw, axis=1, keepdims=True) + 1e-15)
    return Nw


# ================================================================ flat tracer
def trace_flat(prims, o, d, best_t=np.inf):
    N = o.shape[0]
    t_out = np.full(N, INF, np.float64) if np.isscalar(best_t) else best_t.copy()
    prim_idx = np.full(N, -1, np.int32)
    for pi, p in enumerate(prims):
        bc = p.bc
        br = p.br
        wx = o[:, 0] - bc[0]; wy = o[:, 1] - bc[1]; wz = o[:, 2] - bc[2]
        b = wx*d[:, 0] + wy*d[:, 1] + wz*d[:, 2]
        d2 = wx*wx + wy*wy + wz*wz - b*b
        br2 = br*br
        inside = d2 <= br2
        ahead = b + np.sqrt(np.maximum(br2 - d2, 0.0)) > EPS
        near = b - np.sqrt(np.maximum(br2 - d2, 0.0))
        np.maximum(near, EPS, out=near)
        active = (inside | ahead) & (near < t_out)
        sel = np.where(active)[0]
        if len(sel) == 0:
            continue
        o_l, d_l = _local_ray(p, o[sel], d[sel])
        tl = _isect(p, o_l, d_l, t_out[sel])
        m = tl < t_out[sel] - 1e-9
        if not m.any():
            continue
        t_out[sel[m]] = tl[m]
        prim_idx[sel[m]] = pi
    return prim_idx, t_out


def normals_for(prims, o, d, prim_idx, t_out):
    nrm = np.zeros((o.shape[0], 3))
    byp = {}
    for k, pi in enumerate(prim_idx):
        if pi >= 0:
            byp.setdefault(int(pi), []).append(k)
    for pi, ks in byp.items():
        p = prims[pi]
        o_l, d_l = _local_ray(p, o[ks], d[ks])
        nrm[ks] = _normal(p, o_l, d_l, t_out[ks])
    return nrm


# ================================================================ shading
def hemi_amb(nrm, scene):
    t = np.clip(nrm[:, 1]*0.5 + 0.5, 0, 1)
    amb_top = scene.amb_top if scene is not None else np.array([0.3, 0.32, 0.4])
    amb_bot = scene.amb_bot if scene is not None else np.array([0.1, 0.1, 0.12])
    return amb_top[None]*t[:, None] + amb_bot[None]*(1 - t[:, None])


def sky_color(dirs, scene):
    if scene is not None and scene.sky_fn is not None:
        return scene.sky_fn(dirs)
    t = np.clip(dirs[:, 1]*0.5 + 0.5, 0, 1)
    return np.array([0.16, 0.12, 0.12])[None]*(1-t[:, None]) + np.array([0.05, 0.08, 0.16])[None]*t[:, None]


def _prep_mat_arrays(prims):
    P = len(prims)
    albedo = np.stack([p.mat['albedo'] for p in prims]) if P else np.zeros((0, 3))
    rough = np.array([p.mat['rough'] for p in prims]) if P else np.zeros(0)
    refl = np.array([p.mat['refl'] for p in prims]) if P else np.zeros(0)
    spec = np.array([p.mat['spec'] for p in prims]) if P else np.zeros(0)
    sspec = np.array([p.mat['sspec'] for p in prims]) if P else np.zeros(0)
    emis = np.stack([p.mat['emis'] if p.mat['emis'] is not None else np.zeros(3) for p in prims]) if P else np.zeros((0, 3))
    return albedo, rough, refl, spec, sspec, emis


def shade(scene, prims, o, d, prim_idx, t_out, quality=1):
    N = o.shape[0]
    col = np.zeros((N, 3))
    miss = prim_idx < 0
    if miss.any():
        col[miss] = sky_color(d[miss], scene)
    hit = ~miss
    if not hit.any():
        return col
    hs = np.where(hit)[0]
    albA, rgA, rlA, spA, ssA, emA = _prep_mat_arrays(prims)
    gi = prim_idx[hs].astype(np.int64)
    albedo = albA[gi]; rough = rgA[gi]; refl = rlA[gi]
    spec = spA[gi]; sspec = ssA[gi]; emis = emA[gi]
    P = o[hs] + d[hs]*t_out[hs][:, None]
    nrm = normals_for(prims, o[hs], d[hs], prim_idx[hs], t_out[hs])
    V = -d[hs]
    col[hs] = albedo*hemi_amb(nrm, scene) + emis
    shprims = [p for p in prims if not p.mat.get('nocast')]
    for L in scene.lights:
        Lp = np.array(L['pos'], np.float64)
        Lc = np.array(L['color'], np.float64)*L['int']
        Lv = Lp[None] - P
        dist = np.linalg.norm(Lv, axis=1) + 1e-9
        Ln = Lv/dist[:, None]
        ndl = (nrm*Ln).sum(1)
        atten = 1.0/(1.0 + L['radius']*dist**2)
        lit = ndl > 0.0
        if L.get('shadow', True) and lit.any():
            li = np.where(lit)[0]
            so = P[li] + nrm[li]*0.004
            sd = Ln[li]
            s_idx, s_t = trace_flat(shprims, so, sd, dist[li] - 0.01)
            unblocked = s_idx < 0
            newlit = np.zeros_like(lit)
            newlit[li[unblocked]] = True
            lit = newlit
        li = np.where(lit)[0]
        if len(li) == 0:
            continue
        ls = Lc[None]*atten[li][:, None]
        col[hs[li]] += albedo[li]*ndl[li][:, None]*ls
        Hv = Ln[li] + V[li]
        Hv /= (np.linalg.norm(Hv, axis=1, keepdims=True)+1e-15)
        ndh = np.clip((nrm[li]*Hv).sum(1), 0, 1)
        expo = 8.0 + 160.0*np.power(1.0 - rough[li], 3.0)
        spc = np.power(ndh, expo)*(spec[li]*sspec[li]*80.0)
        col[hs[li]] += spc[:, None]*ls
    ndv = np.clip((nrm*V).sum(1), 0, 1)
    rim = (1.0 - ndv)**3.0
    col[hs] += rim[:, None]*0.055*np.array([0.55, 0.7, 0.95])
    rmask = (refl > 0.02) & (quality >= 1)
    if rmask.any():
        rs = np.where(rmask)[0]
        Rd = d[hs][rs] - 2.0*(d[hs][rs]*nrm[rs]).sum(1)[:, None]*nrm[rs]
        Ro = P[rs] + nrm[rs]*0.005
        r_idx, r_t = trace_flat(prims, Ro, Rd, np.full(len(rs), 80.0))
        rcol = sky_color(Rd, scene)
        mh = np.where(r_idx >= 0)[0]
        if len(mh):
            rp = Ro[mh] + Rd[mh]*r_t[mh][:, None]
            rn = normals_for(prims, Ro[mh], Rd[mh], r_idx[mh], r_t[mh])
            byp2 = {}
            for k, pi in zip(mh, r_idx[mh]):
                byp2.setdefault(int(pi), []).append(k)
            for pi, ks in byp2.items():
                m2 = prims[pi].mat
                if m2['emis'] is not None:
                    rcol[ks] += m2['emis']
            ralb = np.stack([prims[_pi].mat['albedo'] for _pi in r_idx[mh]])
            rcol[mh] += ralb*hemi_amb(rn, scene)
            for L in scene.lights:
                Lp = np.array(L['pos'], np.float64)
                Lc = np.array(L['color'], np.float64)*L['int']
                Lv = Lp[None] - rp
                dl = np.linalg.norm(Lv, axis=1) + 1e-9
                Ln = Lv/dl[:, None]
                ndl = np.clip((rn*Ln).sum(1), 0, None)
                atten = 1.0/(1.0 + L['radius']*dl**2)
                rcol[mh] += ralb*ndl[:, None]*Lc[None]*atten[:, None]
        col[hs[rs]] += rcol*refl[rs][:, None]
    if scene.fog is not None:
        fc = np.array(scene.fog['color'], np.float64)
        fagy = 1.0 - np.exp(-t_out[hs]*scene.fog['density'])
        col[hs] = col[hs]*(1-fagy[:, None]) + fc*fagy[:, None]
    return col


# ================================================================ camera
class Camera:
    def __init__(self, eye, target, fov_y=40.0, roll=0.0, dof=None):
        self.eye = np.array(eye, np.float64)
        self.target = np.array(target, np.float64)
        self.roll = roll
        self.fov_y = fov_y
        self.dof = dof
        self.recompute()

    def recompute(self):
        fwd = self.target - self.eye
        fwd = fwd/(np.linalg.norm(fwd)+1e-12)
        up0 = np.array([0, 1.0, 0])
        if abs(fwd @ up0) > 0.999:
            up0 = np.array([0, 0, 1.0])
        right = np.cross(fwd, up0); right /= (np.linalg.norm(right)+1e-12)
        upv = np.cross(right, fwd)
        cr, sr = np.cos(self.roll), np.sin(self.roll)
        self.right = right*cr + upv*sr
        self.upv = upv*cr - right*sr
        self.fwd = fwd

    def rays(self, W, H, jitter=1.0, rng=None):
        ys, xs = np.mgrid[0:H, 0:W]
        xf = xs.ravel().astype(np.float64); yf = ys.ravel().astype(np.float64)
        if jitter and rng is not None:
            xf += rng.random(W*H) - 0.5
            yf += rng.random(W*H) - 0.5
        aspect = W/H
        tanf = np.tan(np.radians(self.fov_y)/2)
        sy = 1.0 - 2.0*(yf + 0.5)/H
        sx = 2.0*(xf + 0.5)/W - 1.0
        sx = sx*aspect*tanf
        sy = sy*tanf
        d = self.fwd[None] + sx[:, None]*self.right[None] + sy[:, None]*self.upv[None]
        d /= (np.linalg.norm(d, axis=1, keepdims=True)+1e-15)
        o = np.broadcast_to(self.eye, d.shape).copy()
        if self.dof is not None:
            focus, ap_strength = self.dof
            r = np.sqrt(rng.random(W*H))*ap_strength
            phi = rng.random(W*H)*2*np.pi
            off = self.right[None]*(np.cos(phi)*r)[:, None] + self.upv[None]*(np.sin(phi)*r)[:, None]
            fwd_dot = (d*self.fwd[None]).sum(1)
            tt = focus/np.maximum(fwd_dot, 0.05)
            fp = o + d*tt[:, None]
            o = o + off
            d = fp - o
            d /= (np.linalg.norm(d, axis=1, keepdims=True)+1e-15)
        return np.ascontiguousarray(o), np.ascontiguousarray(d)


# ================================================================ post
def aces(x):
    a, b, c, dd, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x*(a*x+b))/(x*(c*x+dd)+e), 0, 1)


def bloom(img, thr=1.0, strength=0.5, radius=6):
    from scipy.ndimage import gaussian_filter
    bright = np.clip(img - thr, 0, None)
    if bright.max() <= 0:
        return img
    small = bright[::4, ::4]
    bl = np.stack([gaussian_filter(small[..., k], radius*0.5) for k in range(3)], -1)
    big = np.repeat(np.repeat(bl, 4, 0), 4, 1)[:img.shape[0], :img.shape[1]]
    return img + big*strength


def vignette(img, strength=0.35):
    H, W = img.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W]
    cx, cy = W/2, H/2
    r = np.sqrt(((xs-cx)/(W*0.62))**2 + ((ys-cy)/(H*0.62))**2)
    v = 1.0 - strength*np.clip(r-0.35, 0, None)**1.8
    return img*v[..., None]


def grade(img, exposure=1.0, sat=1.06, contrast=1.03, lift=(0.0, 0.0, 0.01), gain=(1.0, 0.98, 0.965)):
    img = img*exposure
    lum = (img*np.array([0.2126, 0.7152, 0.0722])).sum(-1)
    img = lum[..., None]*(1-sat) + img*sat
    img = (img-0.5)*contrast + 0.5
    img = img*np.array(gain)[None, None] + np.array(lift)[None, None]
    return img


def render_frame(scene, cam, W, H, quality=1, rng=None):
    scene.refresh()
    all_prims = scene.prim_nodes + scene.prim_static
    o, d = cam.rays(W, H, jitter=quality >= 1, rng=rng)
    prim_idx, t_out = trace_flat(all_prims, o, d, best_t=np.inf)
    col = shade(scene, all_prims, o, d, prim_idx, t_out, quality=quality)
    return col, t_out


def finalize(img, exposure=1.0, bloom_strength=0.45, bloom_thr=1.0, vig=0.38):
    img = bloom(img, thr=bloom_thr, strength=bloom_strength)
    img = grade(img, exposure=exposure)
    img = vignette(img, vig)
    return aces(img)


def to_u8(img):
    return (np.clip(img, 0, 1)**(1/2.2)*255).astype(np.uint8)
