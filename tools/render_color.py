#!/usr/bin/env python3
"""Render a colored binary PLY (x,y,z + uchar rgb) as an RGB point cloud, multi-view.
   python tools/render_color.py <ply> <png>"""
import sys, numpy as np
from PIL import Image
def load(path):
    f=open(path,"rb"); assert f.readline().strip()==b"ply"; nv=0; hascol=False
    while True:
        ln=f.readline().strip()
        if ln.startswith(b"element vertex"): nv=int(ln.split()[-1])
        elif ln.startswith(b"property uchar red"): hascol=True
        elif ln==b"end_header": break
    if hascol:
        dt=np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("r","u1"),("g","u1"),("b","u1")])
        a=np.frombuffer(f.read(nv*15),dtype=dt)
        V=np.stack([a["x"],a["y"],a["z"]],1).astype(float); C=np.stack([a["r"],a["g"],a["b"]],1)/255.
    else:
        V=np.frombuffer(f.read(nv*12),"<f4").reshape(nv,3).astype(float); C=np.ones((nv,3))*0.7
    return V,C
V,C=load(sys.argv[1]); R=360; tiles=[]
for ang in [0,90,180,300]:
    a=np.radians(ang); Ry=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    P=V@Ry.T; mn,mx=P.min(0),P.max(0); c=(mn+mx)/2; s=(mx-mn).max()
    x=(((P[:,0]-c[0])/s*0.9+0.5)*R).astype(int); y=((-(P[:,1]-c[1])/s*0.9+0.5)*R).astype(int); z=P[:,2]
    ok=(x>=0)&(x<R)&(y>=0)&(y<R); x,y,z,col=x[ok],y[ok],z[ok],C[ok]
    img=np.zeros((R,R,3)); zb=np.full((R,R),-1e9); o=np.argsort(z)
    sh=((z-z.min())/(z.max()-z.min()+1e-9)*0.5+0.5)
    for i in o:
        if z[i]>zb[y[i],x[i]]: zb[y[i],x[i]]=z[i]; img[y[i],x[i]]=col[i]*sh[i]
    tiles.append(img)
grid=np.concatenate([np.concatenate(tiles[:2],1),np.concatenate(tiles[2:],1)],0)
Image.fromarray((np.clip(grid,0,1)*255).astype(np.uint8)).save(sys.argv[2]); print("saved",sys.argv[2],len(V),"verts")
