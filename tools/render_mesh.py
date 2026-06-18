#!/usr/bin/env python3
"""Tiny headless mesh renderer (numpy software z-buffer, flat shading) for sanity-
checking the generated geometry. Loads our binary PLY, renders a few views to PNG.
    python tools/render_mesh.py <mesh.ply> <out.png>
"""
import sys, struct
import numpy as np

def load_ply(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        nv = nf = 0
        while True:
            ln = f.readline().strip()
            if ln.startswith(b"element vertex"): nv = int(ln.split()[-1])
            elif ln.startswith(b"element face"): nf = int(ln.split()[-1])
            elif ln == b"end_header": break
        V = np.frombuffer(f.read(nv*12), dtype="<f4").reshape(nv, 3).astype(np.float64)
        faces = np.empty((nf, 3), np.int64)
        buf = f.read(nf * 13)   # 1 byte count + 3 int32 per face
        for i in range(nf):
            o = i*13
            faces[i] = struct.unpack_from("<3i", buf, o+1)
    return V, faces

def render(V, F, R, png):
    # rotate to view, orthographic project, z-buffer flat shade
    imgs = []
    for ang in [0, 90, 180, 45]:
        a = np.radians(ang)
        Ry = np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        # tilt down a bit
        t = np.radians(20); Rx = np.array([[1,0,0],[0,np.cos(t),-np.sin(t)],[0,np.sin(t),np.cos(t)]])
        P = V @ Ry.T @ Rx.T
        mn, mx = P.min(0), P.max(0); c = (mn+mx)/2; s = (mx-mn).max()
        px = ((P[:,0]-c[0])/s*0.9+0.5)*R; py = ((-(P[:,1]-c[1]))/s*0.9+0.5)*R; pz = P[:,2]
        img = np.zeros((R,R)); zb = np.full((R,R), -1e9)
        tv = P[F]  # [F,3,3]
        n = np.cross(tv[:,1]-tv[:,0], tv[:,2]-tv[:,0]); ln = np.linalg.norm(n,axis=1)+1e-9
        shade = np.clip(np.abs(n[:,2]/ln), 0.15, 1.0)
        order = np.argsort(tv[:,:,2].mean(1))  # painter-ish, refined by zbuffer
        for fi in order:
            i0,i1,i2 = F[fi]
            xs=[px[i0],px[i1],px[i2]]; ys=[py[i0],py[i1],py[i2]]; zs=[pz[i0],pz[i1],pz[i2]]
            x0=int(max(0,min(xs))); x1=int(min(R-1,max(xs))); y0=int(max(0,min(ys))); y1=int(min(R-1,max(ys)))
            if x1<x0 or y1<y0: continue
            d=( (ys[1]-ys[2])*(xs[0]-xs[2])+(xs[2]-xs[1])*(ys[0]-ys[2]) )
            if abs(d)<1e-6: continue
            for Y in range(y0,y1+1):
                for X in range(x0,x1+1):
                    w0=((ys[1]-ys[2])*(X-xs[2])+(xs[2]-xs[1])*(Y-ys[2]))/d
                    w1=((ys[2]-ys[0])*(X-xs[2])+(xs[0]-xs[2])*(Y-ys[2]))/d
                    w2=1-w0-w1
                    if w0<-0.01 or w1<-0.01 or w2<-0.01: continue
                    z=w0*zs[0]+w1*zs[1]+w2*zs[2]
                    if z>zb[Y,X]: zb[Y,X]=z; img[Y,X]=shade[fi]
        imgs.append(img)
    top=np.concatenate(imgs[:2],1); bot=np.concatenate(imgs[2:],1); grid=np.concatenate([top,bot],0)
    from PIL import Image
    Image.fromarray((np.clip(grid,0,1)*255).astype(np.uint8)).save(png)
    print("saved", png, "V",len(V),"F",len(F))

V,F = load_ply(sys.argv[1])
# subsample faces if huge (for speed)
if len(F) > 280000:
    idx = np.random.default_rng(0).choice(len(F), 120000, replace=False); F = F[idx]
render(V, F, 480, sys.argv[2])
