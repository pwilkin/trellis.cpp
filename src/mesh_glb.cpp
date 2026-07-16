#include "mesh_glb.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cfloat>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef TRELLIS_HAVE_WEBP
#include <webp/encode.h>
#endif

#ifndef TRELLIS_GIT_COMMIT
#define TRELLIS_GIT_COMMIT ""
#endif

#ifndef TRELLIS_GIT_DIRTY
#define TRELLIS_GIT_DIRTY 0
#endif

#ifndef TRELLIS_BUILD_BACKEND
#define TRELLIS_BUILD_BACKEND ""
#endif

#ifndef TRELLIS_BUILD_TIMESTAMP
#define TRELLIS_BUILD_TIMESTAMP ""
#endif

namespace trellis {

static void png_collect(void* ctx, void* data, int size) {
    auto* v = (std::vector<uint8_t>*)ctx;
    v->insert(v->end(), (uint8_t*)data, (uint8_t*)data + size);
}

static void w_u32(std::vector<uint8_t>& o, uint32_t v) {
    o.push_back(v & 0xff); o.push_back((v >> 8) & 0xff);
    o.push_back((v >> 16) & 0xff); o.push_back((v >> 24) & 0xff);
}

static std::string generated_timestamp_utc() {
    std::time_t now = std::time(nullptr);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &now);
#else
    gmtime_r(&now, &tm_utc);
#endif
    char buf[32];
    if (std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc) == 0) {
        return std::string();
    }
    return std::string(buf);
}

static std::string asset_extras_json(int64_t seed) {
    std::string extras;
    const bool have_commit = TRELLIS_GIT_COMMIT[0] != 0;
    const bool have_dirty = TRELLIS_GIT_DIRTY != 0;
    const bool have_backend = TRELLIS_BUILD_BACKEND[0] != 0;
    const bool have_timestamp = TRELLIS_BUILD_TIMESTAMP[0] != 0;
    const std::string generated = generated_timestamp_utc();
    const bool have_generated = !generated.empty();
    if (seed < 0 && !have_commit && !have_dirty && !have_backend && !have_timestamp && !have_generated) {
        return extras;
    }

    extras = ",\"extras\":{";
    bool first = true;
    auto append_sep = [&]() {
        if (!first) {
            extras.push_back(',');
        }
        first = false;
    };

    if (seed >= 0) {
        char seed_buf[64];
        std::snprintf(seed_buf, sizeof(seed_buf), "\"seed\":%lld", (long long) seed);
        append_sep();
        extras += seed_buf;
    }
    if (have_commit) {
        append_sep();
        extras += "\"commit\":\"";
        extras += TRELLIS_GIT_COMMIT;
        extras += "\"";
    }
    if (have_dirty) {
        append_sep();
        extras += "\"dirty\":true";
    }
    if (have_backend) {
        append_sep();
        extras += "\"backend\":\"";
        extras += TRELLIS_BUILD_BACKEND;
        extras += "\"";
    }
    if (have_timestamp) {
        append_sep();
        extras += "\"build\":\"";
        extras += TRELLIS_BUILD_TIMESTAMP;
        extras += "\"";
    }
    if (have_generated) {
        append_sep();
        extras += "\"generated\":\"";
        extras += generated;
        extras += "\"";
    }

    extras += "}";
    return extras;
}

bool write_glb(const char* path, const float* verts, int64_t V, const int32_t* faces, int64_t F,
               const float* colors, int64_t seed) {
    // 1. rotate (x,y,z)->(x,z,-y); track min/max
    std::vector<float> pos((size_t)V * 3);
    float mn[3] = { FLT_MAX, FLT_MAX, FLT_MAX }, mx[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (int64_t i = 0; i < V; ++i) {
        float ox = verts[3*i], oy = verts[3*i+1], oz = verts[3*i+2];
        float p[3] = { ox, oz, -oy };
        for (int c = 0; c < 3; ++c) { pos[3*i+c] = p[c]; mn[c] = std::min(mn[c], p[c]); mx[c] = std::max(mx[c], p[c]); }
    }
    if (V == 0) { mn[0]=mn[1]=mn[2]=0; mx[0]=mx[1]=mx[2]=0; }

    // 2. BIN buffer: positions (V*12), indices (F*12 uint32), [optional colors (V*12 float VEC3)]
    const uint32_t posBytes = (uint32_t)(V * 12), idxBytes = (uint32_t)(F * 12);
    const uint32_t colBytes = colors ? (uint32_t)(V * 12) : 0;
    std::vector<uint8_t> bin(posBytes + idxBytes + colBytes);
    std::memcpy(bin.data(), pos.data(), posBytes);
    for (int64_t i = 0; i < F * 3; ++i) { uint32_t v = (uint32_t)faces[i]; std::memcpy(bin.data() + posBytes + i * 4, &v, 4); }
    if (colors) std::memcpy(bin.data() + posBytes + idxBytes, colors, colBytes);

    // 3. JSON (add COLOR_0 accessor/bufferView when colors present)
    char attr[64], acc2[160], bv2[160];
    if (colors) {
        std::snprintf(attr, sizeof(attr), "\"POSITION\":0,\"COLOR_0\":2");
        std::snprintf(acc2, sizeof(acc2), ",{\"bufferView\":2,\"componentType\":5126,\"count\":%lld,\"type\":\"VEC3\"}", (long long)V);
        std::snprintf(bv2, sizeof(bv2), ",{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962}", posBytes + idxBytes, colBytes);
    } else { attr[0]=0; acc2[0]=0; bv2[0]=0; std::snprintf(attr, sizeof(attr), "\"POSITION\":0"); }

    const std::string asset_meta = asset_extras_json(seed);

    char buf[4096];
    std::snprintf(buf, sizeof(buf),
        "{\"asset\":{\"version\":\"2.0\",\"generator\":\"trellis.cpp\"%s},"
        "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],"
        "\"meshes\":[{\"primitives\":[{\"attributes\":{%s},\"indices\":1,\"mode\":4}]}],"
        "\"accessors\":["
        "{\"bufferView\":0,\"componentType\":5126,\"count\":%lld,\"type\":\"VEC3\","
        "\"min\":[%.9g,%.9g,%.9g],\"max\":[%.9g,%.9g,%.9g]},"
        "{\"bufferView\":1,\"componentType\":5125,\"count\":%lld,\"type\":\"SCALAR\"}%s],"
        "\"bufferViews\":["
        "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":%u,\"target\":34962},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34963}%s],"
        "\"buffers\":[{\"byteLength\":%u}]}",
        asset_meta.c_str(), attr, (long long)V, mn[0], mn[1], mn[2], mx[0], mx[1], mx[2],
        (long long)(F * 3), acc2, posBytes, posBytes, idxBytes, bv2, posBytes + idxBytes + colBytes);
    std::string json(buf);
    while (json.size() % 4 != 0) json.push_back(' ');
    while (bin.size() % 4 != 0) bin.push_back(0);

    uint32_t total = 12 + 8 + (uint32_t)json.size() + 8 + (uint32_t)bin.size();
    std::vector<uint8_t> out;
    w_u32(out, 0x46546C67); w_u32(out, 2); w_u32(out, total);          // header
    w_u32(out, (uint32_t)json.size()); w_u32(out, 0x4E4F534A);         // JSON chunk
    out.insert(out.end(), json.begin(), json.end());
    w_u32(out, (uint32_t)bin.size()); w_u32(out, 0x004E4942);          // BIN chunk
    out.insert(out.end(), bin.begin(), bin.end());

    FILE* f = fopen(path, "wb");
    if (!f) return false;
    bool ok = fwrite(out.data(), 1, out.size(), f) == out.size();
    fclose(f);
    return ok;
}

bool write_glb_textured(const char* path, const float* verts, int64_t V, const float* uv,
                        const int32_t* faces, int64_t F,
                        const unsigned char* base_rgba, const unsigned char* mr_rgba, int T,
                        bool double_sided, int64_t seed) {
    // rotate positions (x,z,-y) + min/max
    std::vector<float> pos((size_t)V*3);
    float mn[3]={FLT_MAX,FLT_MAX,FLT_MAX}, mx[3]={-FLT_MAX,-FLT_MAX,-FLT_MAX};
    for (int64_t i=0;i<V;++i){ float ox=verts[3*i],oy=verts[3*i+1],oz=verts[3*i+2]; float p[3]={ox,oz,-oy};
        for(int c=0;c<3;++c){pos[3*i+c]=p[c]; mn[c]=std::min(mn[c],p[c]); mx[c]=std::max(mx[c],p[c]);} }
    if (V==0){mn[0]=mn[1]=mn[2]=0;mx[0]=mx[1]=mx[2]=0;}

    // area-weighted vertex normals (in the rotated frame) so viewers shade
    // smoothly instead of the glTF-mandated flat fallback
    std::vector<float> nrm((size_t)V*3, 0.f);
    for (int64_t f=0;f<F;++f){
        const int64_t a=faces[3*f], b=faces[3*f+1], c=faces[3*f+2];
        float e1[3], e2[3], n[3];
        for(int k=0;k<3;++k){ e1[k]=pos[3*b+k]-pos[3*a+k]; e2[k]=pos[3*c+k]-pos[3*a+k]; }
        n[0]=e1[1]*e2[2]-e1[2]*e2[1]; n[1]=e1[2]*e2[0]-e1[0]*e2[2]; n[2]=e1[0]*e2[1]-e1[1]*e2[0];
        for(int j=0;j<3;++j){ const int64_t v=faces[3*f+j]; for(int k=0;k<3;++k) nrm[3*v+k]+=n[k]; }
    }
    for (int64_t i=0;i<V;++i){
        const float l=std::sqrt(nrm[3*i]*nrm[3*i]+nrm[3*i+1]*nrm[3*i+1]+nrm[3*i+2]*nrm[3*i+2]);
        if (l>1e-20f) for(int k=0;k<3;++k) nrm[3*i+k]/=l; else nrm[3*i+2]=1.f;
    }

    // encode textures — lossy WebP at the reference's quality when available,
    // PNG otherwise
    std::vector<uint8_t> pngB, pngM;
    bool webp = false;
#ifdef TRELLIS_HAVE_WEBP
    {
        uint8_t* ob = nullptr; uint8_t* om = nullptr;
        const size_t nb = WebPEncodeRGBA(base_rgba, T, T, T*4, 80.f, &ob);
        const size_t nm = WebPEncodeRGBA(mr_rgba, T, T, T*4, 80.f, &om);
        if (nb && nm) {
            pngB.assign(ob, ob + nb);
            pngM.assign(om, om + nm);
            webp = true;
        }
        WebPFree(ob); WebPFree(om);
    }
#endif
    if (!webp) {
        stbi_write_png_to_func(png_collect, &pngB, T, T, 4, base_rgba, T*4);
        stbi_write_png_to_func(png_collect, &pngM, T, T, 4, mr_rgba, T*4);
    }

    auto pad4=[](std::vector<uint8_t>&b){ while(b.size()%4) b.push_back(0); };
    const uint32_t posB=(uint32_t)(V*12), nrmB=(uint32_t)(V*12), uvB=(uint32_t)(V*8), idxB=(uint32_t)(F*12);
    std::vector<uint8_t> bin(posB+nrmB+uvB+idxB);
    std::memcpy(bin.data(), pos.data(), posB);
    std::memcpy(bin.data()+posB, nrm.data(), nrmB);
    std::memcpy(bin.data()+posB+nrmB, uv, uvB);
    for (int64_t i=0;i<F*3;++i){ uint32_t v=(uint32_t)faces[i]; std::memcpy(bin.data()+posB+nrmB+uvB+i*4,&v,4); }
    pad4(bin); const uint32_t off4=(uint32_t)bin.size();
    bin.insert(bin.end(), pngB.begin(), pngB.end()); pad4(bin); const uint32_t off5=(uint32_t)bin.size();
    bin.insert(bin.end(), pngM.begin(), pngM.end()); pad4(bin);

    const std::string asset_meta = asset_extras_json(seed);

    char buf[4096];
    std::snprintf(buf,sizeof(buf),
        "{\"asset\":{\"version\":\"2.0\",\"generator\":\"trellis.cpp\"%s},"
        "%s"
        "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],"
        "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"NORMAL\":1,\"TEXCOORD_0\":2},\"indices\":3,\"material\":0,\"mode\":4}]}],"
        "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorTexture\":{\"index\":0},\"metallicRoughnessTexture\":{\"index\":1},\"metallicFactor\":1.0,\"roughnessFactor\":1.0},\"alphaMode\":\"OPAQUE\",\"doubleSided\":%s}],"
        "%s"
        "\"images\":[{\"bufferView\":4,\"mimeType\":\"%s\"},{\"bufferView\":5,\"mimeType\":\"%s\"}],"
        "\"samplers\":[{}],"
        "\"accessors\":["
        "{\"bufferView\":0,\"componentType\":5126,\"count\":%lld,\"type\":\"VEC3\",\"min\":[%.9g,%.9g,%.9g],\"max\":[%.9g,%.9g,%.9g]},"
        "{\"bufferView\":1,\"componentType\":5126,\"count\":%lld,\"type\":\"VEC3\"},"
        "{\"bufferView\":2,\"componentType\":5126,\"count\":%lld,\"type\":\"VEC2\"},"
        "{\"bufferView\":3,\"componentType\":5125,\"count\":%lld,\"type\":\"SCALAR\"}],"
        "\"bufferViews\":["
        "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":%u,\"target\":34962},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34963},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u}],"
        "\"buffers\":[{\"byteLength\":%u}]}",
        asset_meta.c_str(),
        webp ? "\"extensionsUsed\":[\"EXT_texture_webp\"],\"extensionsRequired\":[\"EXT_texture_webp\"]," : "",
        double_sided ? "true" : "false",
        webp ? "\"textures\":[{\"sampler\":0,\"extensions\":{\"EXT_texture_webp\":{\"source\":0}}},{\"sampler\":0,\"extensions\":{\"EXT_texture_webp\":{\"source\":1}}}],"
             : "\"textures\":[{\"source\":0,\"sampler\":0},{\"source\":1,\"sampler\":0}],",
        webp ? "image/webp" : "image/png", webp ? "image/webp" : "image/png",
        (long long)V, mn[0],mn[1],mn[2],mx[0],mx[1],mx[2],
        (long long)V, (long long)V, (long long)(F*3),
        posB, posB,nrmB, posB+nrmB,uvB, posB+nrmB+uvB,idxB,
        off4,(uint32_t)pngB.size(), off5,(uint32_t)pngM.size(), (uint32_t)bin.size());
    std::string json(buf);
    while (json.size()%4) json.push_back(' ');

    uint32_t total = 12 + 8 + (uint32_t)json.size() + 8 + (uint32_t)bin.size();
    std::vector<uint8_t> o;
    w_u32(o,0x46546C67); w_u32(o,2); w_u32(o,total);
    w_u32(o,(uint32_t)json.size()); w_u32(o,0x4E4F534A); o.insert(o.end(),json.begin(),json.end());
    w_u32(o,(uint32_t)bin.size()); w_u32(o,0x004E4942); o.insert(o.end(),bin.begin(),bin.end());
    FILE* f=fopen(path,"wb"); if(!f) return false;
    bool ok=fwrite(o.data(),1,o.size(),f)==o.size(); fclose(f); return ok;
}

bool write_ply(const char* path, const float* verts, int64_t V, const int32_t* faces, int64_t F,
               const float* colors) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fprintf(f, "ply\nformat binary_little_endian 1.0\n");
    fprintf(f, "element vertex %lld\nproperty float x\nproperty float y\nproperty float z\n", (long long)V);
    if (colors) fprintf(f, "property uchar red\nproperty uchar green\nproperty uchar blue\n");
    fprintf(f, "element face %lld\nproperty list uchar int vertex_indices\nend_header\n", (long long)F);
    for (int64_t i = 0; i < V; ++i) {
        fwrite(verts + 3*i, 4, 3, f);
        if (colors) { unsigned char rgb[3]; for (int c=0;c<3;++c){ float v=colors[3*i+c]*255.f; rgb[c]=(unsigned char)(v<0?0:(v>255?255:v)); } fwrite(rgb, 1, 3, f); }
    }
    for (int64_t i = 0; i < F; ++i) { unsigned char n = 3; fwrite(&n, 1, 1, f); fwrite(faces + 3*i, 4, 3, f); }
    fclose(f);
    return true;
}

} // namespace trellis
