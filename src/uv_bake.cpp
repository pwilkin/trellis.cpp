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

    // 1) occlusion-aware bucket assignment. A face projected along its dominant
    // normal axis may be hidden behind another surface (chin over neck, ear over
    // shoulder); with a shared texel the hidden face samples the occluder's
    // colors, which is the classic box-projection "dirt". Build min/max depth
    // maps per axis from ALL faces (backfaces occlude too), then assign each
    // face to its best axis (by |n|, sign from n) along which it is actually
    // the visible surface. Faces visible along no axis keep their dominant
    // bucket and lose the later depth-tested raster fairly.
    std::vector<float> fnorm((size_t)F * 3);
    float mean_edge = 0.f;
    for (int f = 0; f < F; ++f) {
        const int a = faces[3*f], b = faces[3*f+1], c = faces[3*f+2];
        float e1[3], e2[3];
        for (int k = 0; k < 3; ++k) { e1[k] = verts[3*b+k]-verts[3*a+k]; e2[k] = verts[3*c+k]-verts[3*a+k]; }
        fnorm[3*f+0]=e1[1]*e2[2]-e1[2]*e2[1]; fnorm[3*f+1]=e1[2]*e2[0]-e1[0]*e2[2]; fnorm[3*f+2]=e1[0]*e2[1]-e1[1]*e2[0];
        mean_edge += std::sqrt(e1[0]*e1[0]+e1[1]*e1[1]+e1[2]*e1[2]);
    }
    mean_edge /= (float)F;
    const float eps = 2.f * mean_edge;

    // global projected bbox per axis (depth maps cover the whole mesh)
    float gmin[3][2], gmax[3][2];
    for (int ax = 0; ax < 3; ++ax) { gmin[ax][0]=gmin[ax][1]=1e30f; gmax[ax][0]=gmax[ax][1]=-1e30f; }
    for (int v = 0; v < V; ++v)
        for (int ax = 0; ax < 3; ++ax) {
            const float uu = verts[3*v+UAX[ax]], vv = verts[3*v+VAX[ax]];
            gmin[ax][0]=std::min(gmin[ax][0],uu); gmax[ax][0]=std::max(gmax[ax][0],uu);
            gmin[ax][1]=std::min(gmin[ax][1],vv); gmax[ax][1]=std::max(gmax[ax][1],vv);
        }

    const int D = 1024;
    std::vector<float> dmin((size_t)3*D*D, 1e30f), dmax((size_t)3*D*D, -1e30f);
    auto dtex = [&](int ax, float uu, float vv, float& fx, float& fy) {
        fx = (uu - gmin[ax][0]) / std::max(1e-6f, gmax[ax][0]-gmin[ax][0]) * (D-1);
        fy = (vv - gmin[ax][1]) / std::max(1e-6f, gmax[ax][1]-gmin[ax][1]) * (D-1);
    };
    for (int f = 0; f < F; ++f) {
        const int i0 = faces[3*f], i1 = faces[3*f+1], i2 = faces[3*f+2];
        for (int ax = 0; ax < 3; ++ax) {
            float px[3], py[3], pd[3];
            const int vi[3] = {i0, i1, i2};
            for (int j = 0; j < 3; ++j) {
                dtex(ax, verts[3*vi[j]+UAX[ax]], verts[3*vi[j]+VAX[ax]], px[j], py[j]);
                pd[j] = verts[3*vi[j]+ax];
            }
            int x0=(int)std::floor(std::min({px[0],px[1],px[2]})), x1=(int)std::ceil(std::max({px[0],px[1],px[2]}));
            int y0=(int)std::floor(std::min({py[0],py[1],py[2]})), y1=(int)std::ceil(std::max({py[0],py[1],py[2]}));
            x0=std::max(0,x0); y0=std::max(0,y0); x1=std::min(D-1,x1); y1=std::min(D-1,y1);
            float d=(py[1]-py[2])*(px[0]-px[2])+(px[2]-px[1])*(py[0]-py[2]); if (std::fabs(d)<1e-9f) continue;
            for (int y=y0;y<=y1;++y) for (int x=x0;x<=x1;++x){
                float fx=x+0.5f, fy=y+0.5f;
                float w0=((py[1]-py[2])*(fx-px[2])+(px[2]-px[1])*(fy-py[2]))/d;
                float w1=((py[2]-py[0])*(fx-px[2])+(px[0]-px[2])*(fy-py[2]))/d;
                float w2=1-w0-w1;
                if (w0<-0.05f||w1<-0.05f||w2<-0.05f) continue;
                const float dep = w0*pd[0]+w1*pd[1]+w2*pd[2];
                const size_t t = (size_t)ax*D*D + (size_t)y*D + x;
                dmin[t]=std::min(dmin[t],dep); dmax[t]=std::max(dmax[t],dep);
            }
        }
    }

    auto visible = [&](int f, int ax, int dir)->bool {
        const int vi[3] = {faces[3*f], faces[3*f+1], faces[3*f+2]};
        float cu=0, cv=0, cd=0;
        int pass = 0;
        auto probe = [&](float uu, float vv, float dep)->bool {
            float fx, fy; dtex(ax, uu, vv, fx, fy);
            int x = std::min(D-1, std::max(0, (int)fx)), y = std::min(D-1, std::max(0, (int)fy));
            const size_t t = (size_t)ax*D*D + (size_t)y*D + x;
            return dir == 0 ? (dep >= dmax[t] - eps) : (dep <= dmin[t] + eps);
        };
        for (int j = 0; j < 3; ++j) {
            const float uu=verts[3*vi[j]+UAX[ax]], vv=verts[3*vi[j]+VAX[ax]], dep=verts[3*vi[j]+ax];
            cu+=uu/3; cv+=vv/3; cd+=dep/3;
            if (probe(uu, vv, dep)) ++pass;
        }
        return probe(cu, cv, cd) || pass >= 2;
    };

    // A fallback axis is usable only when the face still has a healthy normal
    // component along it — a near-edge-on face projects to a sliver that
    // rasterizes almost no texels and then samples neighboring charts' colors.
    // Faces hidden along every usable axis (scalp under a raised weapon, chest
    // behind a held prop) go to a second-layer bucket of their dominant axis:
    // stacked surfaces cannot share one projected texel, so the hidden layer
    // gets its own chart. Buckets 0..5 = visible, 6..11 = occluded layer.
    std::vector<int> fbucket((size_t)F);
    int reassigned = 0, layered = 0;
    for (int f = 0; f < F; ++f) {
        const float* n = &fnorm[3*f];
        const float nlen = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        int order[3] = {0, 1, 2};
        std::sort(order, order+3, [&](int a, int b){ return std::fabs(n[a]) > std::fabs(n[b]); });
        const int primary = order[0]*2 + (n[order[0]] >= 0 ? 0 : 1);
        fbucket[f] = 6 + primary;
        for (int k = 0; k < 3; ++k) {
            const int ax = order[k], dir = n[ax] >= 0 ? 0 : 1;
            if (k > 0 && std::fabs(n[ax]) < 0.35f * nlen) break;
            if (visible(f, ax, dir)) { fbucket[f] = ax*2 + dir; if (k > 0) ++reassigned; break; }
        }
        if (fbucket[f] >= 6) ++layered;
    }

    // 2) per-bucket: dedup vertices used, compute projected (u,v) bbox
    const int NB = 12;
    std::vector<int> vmap((size_t)V * NB, -1);            // (vertex,bucket) -> local index
    std::vector<int> bxref[12];                           // local -> original vertex index
    float umin[12], umax[12], vmin[12], vmax[12];
    for (int g = 0; g < NB; ++g) { umin[g]=vmin[g]=1e30f; umax[g]=vmax[g]=-1e30f; }
    auto proj = [&](int v, int g, float& uu, float& vv){ int ax=(g%6)/2; uu=verts[3*v+UAX[ax]]; vv=verts[3*v+VAX[ax]]; };
    auto local = [&](int v, int g)->int {
        int& li = vmap[(size_t)v*NB + g];
        if (li < 0) { li = (int)bxref[g].size(); bxref[g].push_back(v);
            float uu,vv; proj(v,g,uu,vv);
            umin[g]=std::min(umin[g],uu); umax[g]=std::max(umax[g],uu);
            vmin[g]=std::min(vmin[g],vv); vmax[g]=std::max(vmax[g],vv); }
        return li;
    };
    std::vector<std::array<int,3>> bfaces[12];
    for (int f = 0; f < F; ++f) { int g = fbucket[f];
        bfaces[g].push_back({ local(faces[3*f],g), local(faces[3*f+1],g), local(faces[3*f+2],g) }); }

    // 3) atlas layout: 12 buckets (6 visible + 6 occluded-layer) in a
    // 4 (cols) x 3 (rows) grid, aspect-preserving fit + 2px pad
    const int CW = T/4, CH = T/3, pad = 2;
    int base_local[12];                                   // output-vertex offset per bucket
    int Vo = 0; for (int g = 0; g < NB; ++g) { base_local[g] = Vo; Vo += (int)bxref[g].size(); }
    out.verts.resize((size_t)Vo*3); out.uv.resize((size_t)Vo*2);
    std::vector<float> vp(pbr6.empty()?0:(size_t)Vo*6);
    for (int g = 0; g < NB; ++g) {
        const int col = g % 4, row = g / 4;
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
    int Fo = 0; for (int g = 0; g < NB; ++g) Fo += (int)bfaces[g].size();
    out.faces.resize((size_t)Fo*3);
    { int fo = 0; for (int g = 0; g < NB; ++g) for (auto& t : bfaces[g]) {
        out.faces[3*fo+0]=base_local[g]+t[0]; out.faces[3*fo+1]=base_local[g]+t[1]; out.faces[3*fo+2]=base_local[g]+t[2]; ++fo; } }

    // 4) depth-tested rasterization: on residual within-bucket overlap the
    // surface nearest the projection direction owns the texel (outermost wins),
    // instead of whatever face happened to rasterize last.
    out.T = T; out.base.assign((size_t)T*T*4,0); out.mr.assign((size_t)T*T*4,0);
    std::vector<uint8_t> mask((size_t)T*T,0);
    std::vector<float> zbuf((size_t)T*T, -1e30f);
    std::vector<int> fgroup((size_t)Fo);
    { int fo = 0; for (int g = 0; g < NB; ++g) for (size_t i = 0; i < bfaces[g].size(); ++i) fgroup[fo++] = g; }
    auto u8 = [](float v){ v=v*255.f; return (uint8_t)(v<0?0:(v>255?255:v)); };
    for (int f = 0; f < Fo; ++f) {
        int idx[3] = { out.faces[3*f], out.faces[3*f+1], out.faces[3*f+2] };
        const int gax = (fgroup[f]%6)/2;
        const float gsign = (fgroup[f]%2 == 0) ? 1.f : -1.f;
        float px[3], py[3], pz[3];
        for (int j=0;j<3;++j){ px[j]=out.uv[2*idx[j]]*T; py[j]=out.uv[2*idx[j]+1]*T;
            pz[j]=gsign*out.verts[3*idx[j]+gax]; }
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
            const float dep = w0*pz[0]+w1*pz[1]+w2*pz[2];
            if (mask[t] && dep < zbuf[t]) continue;
            zbuf[t] = dep;
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
    printf("  uv_box_project: atlas %dx%d, Vo=%d Fo=%d (6 planes, %d re-bucketed, %d occluded-layer)\n",
           T, T, Vo, Fo, reassigned, layered);
    return out;
}

} // namespace trellis
