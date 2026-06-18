#!/usr/bin/env python3
"""Minimal textured-GLB renderer: parses POSITION/TEXCOORD_0/indices + base-color PNG,
rasterizes with UV texture sampling + z-buffer, multi-view. python tools/render_glb.py <glb> <png>"""
import sys, struct, json, io
import numpy as np
from PIL import Image

def load_glb(path):
    d=open(path,"rb").read()
    assert d[:4]==b"glTF"; n=struct.unpack_from("<I",d,8)[0]
    jl=struct.unpack_from("<I",d,12)[0]; J=json.loads(d[20:20+jl])
    p=20+jl; bl=struct.unpack_from("<I",d,p)[0]; BIN=d[p+8:p+8+bl]
    def acc(i):
        a=J["accessors"][i]; bv=J["bufferViews"][a["bufferView"]]; off=bv.get("byteOffset",0)+a.get("byteOffset",0)
        ct={5126:("<f4",4),5125:("<u4",4),5123:("<u2",2)}[a["componentType"]]
        nc={"SCALAR":1,"VEC2":2,"VEC3":3}[a["type"]]
        arr=np.frombuffer(BIN,dtype=ct[0],count=a["count"]*nc,offset=off).reshape(a["count"],nc)
        return arr
    prim=J["meshes"][0]["primitives"][0]
    P=acc(prim["attributes"]["POSITION"]).astype(float)
    UV=acc(prim["attributes"]["TEXCOORD_0"]).astype(float)
    I=acc(prim["indices"]).reshape(-1,3).astype(int)
    img=J["images"][0]; bv=J["bufferViews"][img["bufferView"]]; o=bv.get("byteOffset",0)
    tex=np.array(Image.open(io.BytesIO(BIN[o:o+bv["byteLength"]])).convert("RGB"))
    return P,UV,I,tex

P,UV,I,tex=load_glb(sys.argv[1]); TH,TW=tex.shape[:2]; R=420
tiles=[]
for ang in [0,90,180,300]:
    a=np.radians(ang); Ry=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    Q=P@Ry.T; mn,mx=Q.min(0),Q.max(0); c=(mn+mx)/2; s=(mx-mn).max()
    sx=((Q[:,0]-c[0])/s*0.9+0.5)*R; sy=(-(Q[:,1]-c[1])/s*0.9+0.5)*R; sz=Q[:,2]
    img=np.zeros((R,R,3),np.uint8); zb=np.full((R,R),-1e9)
    order=np.argsort(Q[I].mean(1)[:,2])
    for fi in order:
        t=I[fi]; xs=sx[t]; ys=sy[t]; zs=sz[t]; uvs=UV[t]
        x0=int(max(0,np.floor(xs.min()))); x1=int(min(R-1,np.ceil(xs.max())))
        y0=int(max(0,np.floor(ys.min()))); y1=int(min(R-1,np.ceil(ys.max())))
        d=(ys[1]-ys[2])*(xs[0]-xs[2])+(xs[2]-xs[1])*(ys[0]-ys[2])
        if abs(d)<1e-9: continue
        for Y in range(y0,y1+1):
            for X in range(x0,x1+1):
                w0=((ys[1]-ys[2])*(X-xs[2])+(xs[2]-xs[1])*(Y-ys[2]))/d
                w1=((ys[2]-ys[0])*(X-xs[2])+(xs[0]-xs[2])*(Y-ys[2]))/d
                w2=1-w0-w1
                if w0<-.01 or w1<-.01 or w2<-.01: continue
                z=w0*zs[0]+w1*zs[1]+w2*zs[2]
                if z<=zb[Y,X]: continue
                zb[Y,X]=z
                u=w0*uvs[0,0]+w1*uvs[1,0]+w2*uvs[2,0]; v=w0*uvs[0,1]+w1*uvs[1,1]+w2*uvs[2,1]
                tx=min(TW-1,max(0,int(u*TW))); ty=min(TH-1,max(0,int(v*TH)))
                sh=0.5+0.5*max(0,z/ (abs(zs).max()+1e-9))
                img[Y,X]=(tex[ty,tx]*sh).astype(np.uint8)
    tiles.append(img)
grid=np.concatenate([np.concatenate(tiles[:2],1),np.concatenate(tiles[2:],1)],0)
Image.fromarray(grid).save(sys.argv[2]); print("saved",sys.argv[2],"verts",len(P),"faces",len(I),"tex",tex.shape)
