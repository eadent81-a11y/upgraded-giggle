import numpy as np, time
H, W = 480, 270   # small test, will scale
npix = H*W
rays_o = np.random.rand(npix,3).astype(np.float32)*np.array([1,1,0.1]) + np.array([-1,-1,5])
rays_d = np.random.randn(npix,3).astype(np.float32); rays_d /= np.linalg.norm(rays_d,axis=1,keepdims=True)

N = 120
pc = np.random.rand(N,3).astype(np.float32)*4 + np.array([-2,0,-2])
pr = (0.05+np.random.rand(N)*0.1).astype(np.float32)

t0=time.time()
best_t = np.full(npix, 1e9, np.float32)
mask = np.ones(npix, bool)
l2 = None
for i in range(N):
    o = rays_o; d = rays_d
    oc = o - pc[i]
    a = (d*d).sum(1); b = 2*(oc*d).sum(1); c = (oc*oc).sum(1)-pr[i]*pr[i]
    disc = b*b-4*a*c
    ok = (disc>0)
    sq = np.sqrt(np.maximum(disc,0))
    t = (-b-sq)/(2*a+1e-12)
    hit = ok & (t>0.01) & (t<best_t)
    best_t[hit] = t[hit]
print(f" {N} spheres: {time.time()-t0:.3f}s for {npix}px -> {(time.time()-t0)/npix*1e6:.2f} us/px")
est_full = 518400/npix * (time.time()-t0)
print(f" estimated 540x960 core: {est_full:.2f}s/frame (no shadows)")
t0=time.time()
for i in range(N):
    oc = rays_o - pc[i]
    a=(rays_d*rays_d).sum(1); b=2*(oc*rays_d).sum(1); c0=(oc*oc).sum(1)-pr[i]*pr[i]
    disc=b*b-4*a*c0; ok=disc>0; sq=np.sqrt(np.maximum(disc,0)); t=(-b-sq)/(2*a+1e-12)
    sh = ok&(t>0.01)&(t<20)
print(f" + shadow pass all prims: {time.time()-t0:.3f}s ({(time.time()-t0)/npix*1e6:.2f} us/px)")
