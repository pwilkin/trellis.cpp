#include "uv_bake.h"
#include "xatlas.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <array>

namespace trellis {

void decimate_cluster(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                      const std::vector<float>& pbr6, int grid,
                      std::vector<float>& ov, std::vector<int32_t>& of, std::vector<float>& op) {
    const bool hp = !pbr6.empty();
    auto cell = [&](int v, int a){ int c = (int)((verts[3*v+a] + 0.5f) * grid); return c < 0 ? 0 : (c >= grid ? grid-1 : c); };
    auto key = [](int x,int y,int z){ return ((uint64_t)(uint32_t)x<<40)|((uint64_t)(uint32_t)y<<20)|(uint32_t)z; };
    std::unordered_map<uint64_t,int> cmap; cmap.reserve(V);
    std::vector<int> vmap(V);
    std::vector<double> psum, csum; std::vector<int> cnt;
    for (int v = 0; v < V; ++v) {
        uint64_t k = key(cell(v,0), cell(v,1), cell(v,2));
        auto it = cmap.find(k);
        int idx;
        if (it == cmap.end()) { idx = (int)cnt.size(); cmap[k] = idx; cnt.push_back(0); psum.insert(psum.end(),{0,0,0}); if(hp) csum.insert(csum.end(),{0,0,0,0,0,0}); }
        else idx = it->second;
        vmap[v] = idx; cnt[idx]++;
        for (int a=0;a<3;++a) psum[3*idx+a] += verts[3*v+a];
        if (hp) for (int a=0;a<6;++a) csum[6*idx+a] += pbr6[6*v+a];
    }
    const int M = (int)cnt.size();
    ov.resize((size_t)M*3); if(hp) op.resize((size_t)M*6);
    for (int i=0;i<M;++i){ for(int a=0;a<3;++a) ov[3*i+a]=(float)(psum[3*i+a]/cnt[i]); if(hp) for(int a=0;a<6;++a) op[6*i+a]=(float)(csum[6*i+a]/cnt[i]); }
    of.clear(); of.reserve((size_t)F*3);
    for (int f=0;f<F;++f){ int a=vmap[faces[3*f]],b=vmap[faces[3*f+1]],c=vmap[faces[3*f+2]];
        if (a!=b && b!=c && a!=c) { of.push_back(a); of.push_back(b); of.push_back(c); } }
    printf("  decimate(grid=%d): V %d->%d, F %d->%d\n", grid, V, M, F, (int)of.size()/3);
}

BakedMesh uv_bake(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                  const std::vector<float>& pbr6, int texsize) {
    BakedMesh out;

    // --- xatlas unwrap ---
    xatlas::Atlas* atlas = xatlas::Create();
    xatlas::MeshDecl md;
    md.vertexCount = (uint32_t)V;
    md.vertexPositionData = verts.data();
    md.vertexPositionStride = 3 * sizeof(float);
    md.indexCount = (uint32_t)F * 3;
    md.indexData = faces.data();
    md.indexFormat = xatlas::IndexFormat::UInt32;
    if (xatlas::AddMesh(atlas, md) != xatlas::AddMeshError::Success) { xatlas::Destroy(atlas); return out; }
    xatlas::ChartOptions co;            // defaults
    xatlas::PackOptions po; po.resolution = (uint32_t)texsize; po.padding = 2; po.bilinear = true; po.blockAlign = true;
    xatlas::Generate(atlas, co, po);
    if (atlas->meshCount == 0 || atlas->width == 0) { xatlas::Destroy(atlas); return out; }

    const xatlas::Mesh& m = atlas->meshes[0];
    const int W = (int)atlas->width, H = (int)atlas->height, T = std::max(W, H);
    const int Vo = (int)m.vertexCount, Fo = (int)m.indexCount / 3;

    // --- re-indexed atlas vertices (pos, uv, pbr from xref) ---
    out.verts.resize((size_t)Vo * 3);
    out.uv.resize((size_t)Vo * 2);
    std::vector<float> vp(pbr6.empty() ? 0 : (size_t)Vo * 6);
    for (int i = 0; i < Vo; ++i) {
        uint32_t xref = m.vertexArray[i].xref;
        for (int k = 0; k < 3; ++k) out.verts[3*i+k] = verts[3*xref+k];
        out.uv[2*i+0] = m.vertexArray[i].uv[0] / W;
        out.uv[2*i+1] = m.vertexArray[i].uv[1] / H;
        if (!pbr6.empty()) for (int k = 0; k < 6; ++k) vp[6*i+k] = pbr6[6*xref+k];
    }
    out.faces.assign(m.indexArray, m.indexArray + (size_t)Fo * 3);

    // --- rasterize per-vertex PBR into the atlas (texel coords = uv * W,H) ---
    out.T = T;
    out.base.assign((size_t)T * T * 4, 0);
    out.mr.assign((size_t)T * T * 4, 0);
    std::vector<uint8_t> mask((size_t)T * T, 0);
    auto u8 = [](float v){ v = v*255.f; return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); };
    for (int f = 0; f < Fo; ++f) {
        int idx[3] = { (int)out.faces[3*f], (int)out.faces[3*f+1], (int)out.faces[3*f+2] };
        float px[3], py[3];
        for (int j = 0; j < 3; ++j) { px[j] = out.uv[2*idx[j]] * W; py[j] = out.uv[2*idx[j]+1] * H; }
        int x0 = (int)std::floor(std::min({px[0],px[1],px[2]})), x1 = (int)std::ceil(std::max({px[0],px[1],px[2]}));
        int y0 = (int)std::floor(std::min({py[0],py[1],py[2]})), y1 = (int)std::ceil(std::max({py[0],py[1],py[2]}));
        x0 = std::max(0,x0); y0 = std::max(0,y0); x1 = std::min(T-1,x1); y1 = std::min(T-1,y1);
        float d = (py[1]-py[2])*(px[0]-px[2]) + (px[2]-px[1])*(py[0]-py[2]);
        if (std::fabs(d) < 1e-9f) continue;
        for (int y = y0; y <= y1; ++y) for (int x = x0; x <= x1; ++x) {
            float fx = x + 0.5f, fy = y + 0.5f;
            float w0 = ((py[1]-py[2])*(fx-px[2]) + (px[2]-px[1])*(fy-py[2]))/d;
            float w1 = ((py[2]-py[0])*(fx-px[2]) + (px[0]-px[2])*(fy-py[2]))/d;
            float w2 = 1 - w0 - w1;
            if (w0 < -0.001f || w1 < -0.001f || w2 < -0.001f) continue;
            size_t t = (size_t)y * T + x;
            if (!pbr6.empty()) {
                float br = w0*vp[6*idx[0]+0]+w1*vp[6*idx[1]+0]+w2*vp[6*idx[2]+0];
                float bg = w0*vp[6*idx[0]+1]+w1*vp[6*idx[1]+1]+w2*vp[6*idx[2]+1];
                float bb = w0*vp[6*idx[0]+2]+w1*vp[6*idx[1]+2]+w2*vp[6*idx[2]+2];
                float met= w0*vp[6*idx[0]+3]+w1*vp[6*idx[1]+3]+w2*vp[6*idx[2]+3];
                float rgh= w0*vp[6*idx[0]+4]+w1*vp[6*idx[1]+4]+w2*vp[6*idx[2]+4];
                out.base[4*t+0]=u8(br); out.base[4*t+1]=u8(bg); out.base[4*t+2]=u8(bb); out.base[4*t+3]=255;
                out.mr[4*t+0]=255; out.mr[4*t+1]=u8(rgh); out.mr[4*t+2]=u8(met); out.mr[4*t+3]=255;
            }
            mask[t] = 1;
        }
    }
    xatlas::Destroy(atlas);

    // --- seam dilation: spread written texels into unwritten neighbors (a few passes) ---
    for (int pass = 0; pass < 6; ++pass) {
        std::vector<uint8_t> nm = mask;
        for (int y = 0; y < T; ++y) for (int x = 0; x < T; ++x) {
            size_t t = (size_t)y*T+x;
            if (mask[t]) continue;
            for (int dy=-1; dy<=1 && !nm[t]; ++dy) for (int dx=-1; dx<=1; ++dx) {
                int xx=x+dx, yy=y+dy; if (xx<0||yy<0||xx>=T||yy>=T) continue;
                size_t s=(size_t)yy*T+xx; if (!mask[s]) continue;
                for (int c=0;c<4;++c){ out.base[4*t+c]=out.base[4*s+c]; out.mr[4*t+c]=out.mr[4*s+c]; }
                nm[t]=1; break;
            }
        }
        mask.swap(nm);
    }
    printf("  uv_bake: atlas %dx%d, Vo=%d Fo=%d\n", W, H, Vo, Fo);
    return out;
}

BakedMesh uv_box_project(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                         const std::vector<float>& pbr6, int texsize) {
    BakedMesh out;
    const int T = texsize;
    if (F == 0) return out;

    // in-plane axes for each dominant axis: X->(Y,Z), Y->(X,Z), Z->(X,Y)
    static const int UAX[3] = {1, 0, 0}, VAX[3] = {2, 2, 1};

    // 1) classify each face into one of 6 buckets by its dominant signed normal
    std::vector<int> fbucket((size_t)F);
    for (int f = 0; f < F; ++f) {
        const int a = faces[3*f], b = faces[3*f+1], c = faces[3*f+2];
        float e1[3], e2[3], n[3];
        for (int k = 0; k < 3; ++k) { e1[k] = verts[3*b+k]-verts[3*a+k]; e2[k] = verts[3*c+k]-verts[3*a+k]; }
        n[0]=e1[1]*e2[2]-e1[2]*e2[1]; n[1]=e1[2]*e2[0]-e1[0]*e2[2]; n[2]=e1[0]*e2[1]-e1[1]*e2[0];
        int ax = 0; float m0 = std::fabs(n[0]);
        if (std::fabs(n[1]) > m0) { ax = 1; m0 = std::fabs(n[1]); }
        if (std::fabs(n[2]) > m0) { ax = 2; }
        fbucket[f] = ax*2 + (n[ax] >= 0 ? 0 : 1);
    }

    // 2) per-bucket: dedup vertices used, compute projected (u,v) bbox
    std::vector<int> vmap((size_t)V * 6, -1);             // (vertex,bucket) -> local index
    std::vector<int> bxref[6];                            // local -> original vertex index
    float umin[6], umax[6], vmin[6], vmax[6];
    for (int g = 0; g < 6; ++g) { umin[g]=vmin[g]=1e30f; umax[g]=vmax[g]=-1e30f; }
    auto proj = [&](int v, int g, float& uu, float& vv){ int ax=g/2; uu=verts[3*v+UAX[ax]]; vv=verts[3*v+VAX[ax]]; };
    auto local = [&](int v, int g)->int {
        int& li = vmap[(size_t)v*6 + g];
        if (li < 0) { li = (int)bxref[g].size(); bxref[g].push_back(v);
            float uu,vv; proj(v,g,uu,vv);
            umin[g]=std::min(umin[g],uu); umax[g]=std::max(umax[g],uu);
            vmin[g]=std::min(vmin[g],vv); vmax[g]=std::max(vmax[g],vv); }
        return li;
    };
    std::vector<std::array<int,3>> bfaces[6];
    for (int f = 0; f < F; ++f) { int g = fbucket[f];
        bfaces[g].push_back({ local(faces[3*f],g), local(faces[3*f+1],g), local(faces[3*f+2],g) }); }

    // 3) atlas layout: 6 buckets in a 3 (cols) x 2 (rows) grid, aspect-preserving fit + 2px pad
    const int CW = T/3, CH = T/2, pad = 2;
    int base_local[6];                                    // output-vertex offset per bucket
    int Vo = 0; for (int g = 0; g < 6; ++g) { base_local[g] = Vo; Vo += (int)bxref[g].size(); }
    out.verts.resize((size_t)Vo*3); out.uv.resize((size_t)Vo*2);
    std::vector<float> vp(pbr6.empty()?0:(size_t)Vo*6);
    for (int g = 0; g < 6; ++g) {
        const int col = g % 3, row = g / 3;
        const float ox = (float)col*CW + pad, oy = (float)row*CH + pad;
        const float bw = std::max(1e-6f, umax[g]-umin[g]), bh = std::max(1e-6f, vmax[g]-vmin[g]);
        const float s = std::min((CW-2*pad)/bw, (CH-2*pad)/bh);   // uniform scale, no stretch
        for (int i = 0; i < (int)bxref[g].size(); ++i) {
            const int v = bxref[g][i], o = base_local[g] + i;
            for (int k = 0; k < 3; ++k) out.verts[3*o+k] = verts[3*v+k];
            float uu,vv; proj(v,g,uu,vv);
            out.uv[2*o+0] = (ox + (uu-umin[g])*s) / T;
            out.uv[2*o+1] = (oy + (vv-vmin[g])*s) / T;
            if (!pbr6.empty()) for (int k = 0; k < 6; ++k) vp[6*o+k] = pbr6[6*v+k];
        }
    }
    int Fo = 0; for (int g = 0; g < 6; ++g) Fo += (int)bfaces[g].size();
    out.faces.resize((size_t)Fo*3);
    { int fo = 0; for (int g = 0; g < 6; ++g) for (auto& t : bfaces[g]) {
        out.faces[3*fo+0]=base_local[g]+t[0]; out.faces[3*fo+1]=base_local[g]+t[1]; out.faces[3*fo+2]=base_local[g]+t[2]; ++fo; } }

    // 4) rasterize per-vertex PBR into the atlas (last-wins on within-bucket overlap)
    out.T = T; out.base.assign((size_t)T*T*4,0); out.mr.assign((size_t)T*T*4,0);
    std::vector<uint8_t> mask((size_t)T*T,0);
    auto u8 = [](float v){ v=v*255.f; return (uint8_t)(v<0?0:(v>255?255:v)); };
    for (int f = 0; f < Fo; ++f) {
        int idx[3] = { out.faces[3*f], out.faces[3*f+1], out.faces[3*f+2] };
        float px[3], py[3];
        for (int j=0;j<3;++j){ px[j]=out.uv[2*idx[j]]*T; py[j]=out.uv[2*idx[j]+1]*T; }
        int x0=(int)std::floor(std::min({px[0],px[1],px[2]})), x1=(int)std::ceil(std::max({px[0],px[1],px[2]}));
        int y0=(int)std::floor(std::min({py[0],py[1],py[2]})), y1=(int)std::ceil(std::max({py[0],py[1],py[2]}));
        x0=std::max(0,x0); y0=std::max(0,y0); x1=std::min(T-1,x1); y1=std::min(T-1,y1);
        float d=(py[1]-py[2])*(px[0]-px[2])+(px[2]-px[1])*(py[0]-py[2]); if (std::fabs(d)<1e-9f) continue;
        for (int y=y0;y<=y1;++y) for (int x=x0;x<=x1;++x){
            float fx=x+0.5f, fy=y+0.5f;
            float w0=((py[1]-py[2])*(fx-px[2])+(px[2]-px[1])*(fy-py[2]))/d;
            float w1=((py[2]-py[0])*(fx-px[2])+(px[0]-px[2])*(fy-py[2]))/d;
            float w2=1-w0-w1;
            if (w0<-0.001f||w1<-0.001f||w2<-0.001f) continue;
            size_t t=(size_t)y*T+x;
            if (!pbr6.empty()){
                float br=w0*vp[6*idx[0]+0]+w1*vp[6*idx[1]+0]+w2*vp[6*idx[2]+0];
                float bg=w0*vp[6*idx[0]+1]+w1*vp[6*idx[1]+1]+w2*vp[6*idx[2]+1];
                float bb=w0*vp[6*idx[0]+2]+w1*vp[6*idx[1]+2]+w2*vp[6*idx[2]+2];
                float met=w0*vp[6*idx[0]+3]+w1*vp[6*idx[1]+3]+w2*vp[6*idx[2]+3];
                float rgh=w0*vp[6*idx[0]+4]+w1*vp[6*idx[1]+4]+w2*vp[6*idx[2]+4];
                out.base[4*t+0]=u8(br); out.base[4*t+1]=u8(bg); out.base[4*t+2]=u8(bb); out.base[4*t+3]=255;
                out.mr[4*t+0]=255; out.mr[4*t+1]=u8(rgh); out.mr[4*t+2]=u8(met); out.mr[4*t+3]=255;
            }
            mask[t]=1;
        }
    }
    // 5) seam dilation
    for (int pass=0; pass<4; ++pass){
        std::vector<uint8_t> nm=mask;
        for (int y=0;y<T;++y) for (int x=0;x<T;++x){ size_t t=(size_t)y*T+x; if(mask[t])continue;
            for (int dy=-1;dy<=1&&!nm[t];++dy) for (int dx=-1;dx<=1;++dx){ int xx=x+dx,yy=y+dy;
                if(xx<0||yy<0||xx>=T||yy>=T)continue; size_t s=(size_t)yy*T+xx; if(!mask[s])continue;
                for(int c=0;c<4;++c){ out.base[4*t+c]=out.base[4*s+c]; out.mr[4*t+c]=out.mr[4*s+c]; } nm[t]=1; break; } }
        mask.swap(nm);
    }
    printf("  uv_box_project: atlas %dx%d, Vo=%d Fo=%d (6 planes)\n", T, T, Vo, Fo);
    return out;
}

} // namespace trellis
