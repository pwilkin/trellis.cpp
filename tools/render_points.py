#!/usr/bin/env python3
"""Dense point-cloud render of mesh vertices (vectorized z-buffer splat) — fast,
clear silhouette for sanity-checking shape. python tools/render_points.py <ply> <png>"""
import sys, struct
import numpy as np
from PIL import Image

def load_ply_verts(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        nv = 0
        while True:
            ln = f.readline().strip()
            if ln.startswith(b"element vertex"): nv = int(ln.split()[-1])
            elif ln == b"end_header": break
        return np.frombuffer(f.read(nv*12), dtype="<f4").reshape(nv,3).astype(np.float64)

V = load_ply_verts(sys.argv[1]); R = 320
views = {0:0, 1:90, 2:180, 3:270}
imgs = []
for ang in [0,90,180,45]:
    a=np.radians(ang); Ry=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    t=np.radians(15); Rx=np.array([[1,0,0],[0,np.cos(t),-np.sin(t)],[0,np.sin(t),np.cos(t)]])
    P = V @ Ry.T @ Rx.T
    mn,mx=P.min(0),P.max(0); c=(mn+mx)/2; s=(mx-mn).max()
    x=(((P[:,0]-c[0])/s*0.85+0.5)*R).astype(int); y=((-(P[:,1]-c[1])/s*0.85+0.5)*R).astype(int); z=P[:,2]
    ok=(x>=0)&(x<R)&(y>=0)&(y<R); x,y,z=x[ok],y[ok],z[ok]
    order=np.argsort(z)  # far first, near overwrites
    img=np.zeros((R,R))
    sh=(z-z.min())/(z.max()-z.min()+1e-9)*0.8+0.2
    img[y[order],x[order]]=sh[order]
    # 1px dilation for visibility
    d=img.copy()
    for dy in(-1,0,1):
      for dx in(-1,0,1):
        d[max(0,dy):R+min(0,dy),max(0,dx):R+min(0,dx)]=np.maximum(d[max(0,dy):R+min(0,dy),max(0,dx):R+min(0,dx)], img[max(0,-dy):R-max(0,dy),max(0,-dx):R-max(0,dx)])
    imgs.append(d)
grid=np.concatenate([np.concatenate(imgs[:2],1),np.concatenate(imgs[2:],1)],0)
Image.fromarray((np.clip(grid,0,1)*255).astype(np.uint8)).save(sys.argv[2])
print("saved",sys.argv[2],"verts",len(V))
