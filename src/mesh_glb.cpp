#include "mesh_glb.h"
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cfloat>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <array>
#include <unordered_map>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

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

#ifndef TRELLIS_APP_VERSION
#define TRELLIS_APP_VERSION ""
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

static void appendf(std::string& s, const char* fmt, ...) {
    char buf[2048];
    va_list ap; va_start(ap, fmt);
    const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n > 0) s.append(buf, (size_t)std::min(n, (int)sizeof(buf) - 1));
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

static std::string generator_label() {
    if (TRELLIS_APP_VERSION[0] == 0) {
        return "trellis.cpp";
    }
    std::string g = "trellis.cpp ";
    g += TRELLIS_APP_VERSION;
    return g;
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char ch : s) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (ch < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned int)ch);
                    out += buf;
                } else {
                    out.push_back((char)ch);
                }
                break;
        }
    }
    return out;
}

static std::string normalize_cli_text_utf8(const char* s) {
    if (!s || !s[0]) {
        return std::string();
    }
#if defined(_WIN32)
    int wlen = MultiByteToWideChar(CP_ACP, 0, s, -1, nullptr, 0);
    if (wlen <= 1) {
        return std::string(s);
    }

    std::wstring w((size_t) wlen, L'\0');
    if (MultiByteToWideChar(CP_ACP, 0, s, -1, &w[0], wlen) <= 0) {
        return std::string(s);
    }

    int u8len = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (u8len <= 1) {
        return std::string(s);
    }

    std::string u8((size_t) u8len, '\0');
    if (WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, &u8[0], u8len, nullptr, nullptr) <= 0) {
        return std::string(s);
    }
    if (!u8.empty() && u8.back() == '\0') {
        u8.pop_back();
    }
    return u8;
#else
    return std::string(s);
#endif
}

static std::string asset_copyright_json(const char* copyright) {
    if (!copyright || !copyright[0]) {
        return std::string();
    }
    std::string s = ",\"copyright\":\"";
    s += json_escape(normalize_cli_text_utf8(copyright));
    s += "\"";
    return s;
}

bool write_glb(const char* path, const float* verts, int64_t V, const int32_t* faces, int64_t F,
               const float* colors, int64_t seed, const char* copyright) {
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
    const std::string generator = generator_label();
    const std::string asset_copyright = asset_copyright_json(copyright);

    char buf[4096];
    std::snprintf(buf, sizeof(buf),
        "{\"asset\":{\"version\":\"2.0\",\"minVersion\":\"2.0\",\"generator\":\"%s\"%s%s},"
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
        generator.c_str(), asset_copyright.c_str(), asset_meta.c_str(), attr, (long long)V, mn[0], mn[1], mn[2], mx[0], mx[1], mx[2],
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

std::vector<float> welded_normals_gltf(const float* pos, int64_t V,
                                       const int32_t* faces, int64_t F) {
    std::vector<int64_t> rep((size_t)V);
    {
        struct KeyHash {
            size_t operator()(const std::array<float,3>& k) const {
                size_t h = 1469598103934665603ull;
                const unsigned char* p = (const unsigned char*)k.data();
                for (int i = 0; i < 12; ++i) { h ^= p[i]; h *= 1099511628211ull; }
                return h;
            }
        };
        std::unordered_map<std::array<float,3>, int64_t, KeyHash> first;
        first.reserve((size_t)V);
        for (int64_t i=0;i<V;++i){
            const std::array<float,3> k{pos[3*i], pos[3*i+1], pos[3*i+2]};
            rep[i] = first.emplace(k, i).first->second;
        }
    }
    std::vector<float> nrm((size_t)V*3, 0.f);
    for (int64_t f=0;f<F;++f){
        const int64_t a=faces[3*f], b=faces[3*f+1], c=faces[3*f+2];
        float e1[3], e2[3], n[3];
        for(int k=0;k<3;++k){ e1[k]=pos[3*b+k]-pos[3*a+k]; e2[k]=pos[3*c+k]-pos[3*a+k]; }
        n[0]=e1[1]*e2[2]-e1[2]*e2[1]; n[1]=e1[2]*e2[0]-e1[0]*e2[2]; n[2]=e1[0]*e2[1]-e1[1]*e2[0];
        for(int j=0;j<3;++j){ const int64_t v=rep[faces[3*f+j]]; for(int k=0;k<3;++k) nrm[3*v+k]+=n[k]; }
    }
    for (int64_t i=0;i<V;++i){
        if (rep[i] != i) continue;
        const float l=std::sqrt(nrm[3*i]*nrm[3*i]+nrm[3*i+1]*nrm[3*i+1]+nrm[3*i+2]*nrm[3*i+2]);
        if (l>1e-20f) for(int k=0;k<3;++k) nrm[3*i+k]/=l; else nrm[3*i+2]=1.f;
    }
    for (int64_t i=0;i<V;++i){
        const int64_t r=rep[i];
        if (r != i) for(int k=0;k<3;++k) nrm[3*i+k]=nrm[3*r+k];
    }
    return nrm;
}

bool write_glb_textured(const char* path, const float* verts, int64_t V, const float* uv,
                        const int32_t* faces, int64_t F,
                        const unsigned char* base_rgba, const unsigned char* mr_rgba, int T,
                        bool double_sided, int64_t seed, const char* copyright, bool use_webp,
                        const unsigned char* nrm_rgba, const float* tangents) {
    // rotate positions (x,z,-y) + min/max
    std::vector<float> pos((size_t)V*3);
    float mn[3]={FLT_MAX,FLT_MAX,FLT_MAX}, mx[3]={-FLT_MAX,-FLT_MAX,-FLT_MAX};
    for (int64_t i=0;i<V;++i){ float ox=verts[3*i],oy=verts[3*i+1],oz=verts[3*i+2]; float p[3]={ox,oz,-oy};
        for(int c=0;c<3;++c){pos[3*i+c]=p[c]; mn[c]=std::min(mn[c],p[c]); mx[c]=std::max(mx[c],p[c]);} }
    if (V==0){mn[0]=mn[1]=mn[2]=0;mx[0]=mx[1]=mx[2]=0;}
    const auto nrm=welded_normals_gltf(pos.data(),V,faces,F);

    // encode textures — lossy WebP at the reference's quality when available,
    // PNG otherwise
    std::vector<uint8_t> pngB, pngM;
    bool webp = false;
#ifdef TRELLIS_HAVE_WEBP
    if (use_webp) {
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
    // Optional tangent-space normal map: a third image plus a TANGENT accessor so
    // viewers use the exact frame the bake used. Always PNG — lossy WebP on a normal
    // map shows up as shading noise — which is valid alongside EXT_texture_webp.
    const bool has_nrm = nrm_rgba != nullptr && tangents != nullptr;
    std::vector<uint8_t> pngN;
    if (has_nrm) stbi_write_png_to_func(png_collect, &pngN, T, T, 4, nrm_rgba, T*4);
    const uint32_t posB=(uint32_t)(V*12), nrmB=(uint32_t)(V*12), uvB=(uint32_t)(V*8),
                   tanB=has_nrm ? (uint32_t)(V*16) : 0u, idxB=(uint32_t)(F*12);
    std::vector<uint8_t> bin(posB+nrmB+uvB+tanB+idxB);
    std::memcpy(bin.data(), pos.data(), posB);
    std::memcpy(bin.data()+posB, nrm.data(), nrmB);
    std::memcpy(bin.data()+posB+nrmB, uv, uvB);
    if (has_nrm) {   // same TRELLIS->glTF rotation as the positions; w unchanged
        float* tan = (float*)(bin.data()+posB+nrmB+uvB);
        for (int64_t i=0;i<V;++i){ tan[4*i]=tangents[4*i]; tan[4*i+1]=tangents[4*i+2]; tan[4*i+2]=-tangents[4*i+1]; tan[4*i+3]=tangents[4*i+3]; }
    }
    const uint32_t idxOff = posB+nrmB+uvB+tanB;
    for (int64_t i=0;i<F*3;++i){ uint32_t v=(uint32_t)faces[i]; std::memcpy(bin.data()+idxOff+i*4,&v,4); }
    pad4(bin); const uint32_t offB=(uint32_t)bin.size();
    bin.insert(bin.end(), pngB.begin(), pngB.end()); pad4(bin); const uint32_t offM=(uint32_t)bin.size();
    bin.insert(bin.end(), pngM.begin(), pngM.end()); pad4(bin); const uint32_t offN=(uint32_t)bin.size();
    if (has_nrm) { bin.insert(bin.end(), pngN.begin(), pngN.end()); pad4(bin); }

    const std::string asset_meta = asset_extras_json(seed);
    const std::string generator = generator_label();
    const std::string asset_copyright = asset_copyright_json(copyright);

    // bufferViews: 0 POSITION, 1 NORMAL, 2 TEXCOORD_0, [3 TANGENT], then indices, then images.
    const int bvIdx = has_nrm ? 4 : 3, bvImg = bvIdx + 1;
    const char* mime = webp ? "image/webp" : "image/png";
    std::string json;
    json.reserve(8192);
    appendf(json,
        "{\"asset\":{\"version\":\"2.0\",\"minVersion\":\"2.0\",\"generator\":\"%s\"%s%s},"
        "%s"
        "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],",
        generator.c_str(), asset_copyright.c_str(), asset_meta.c_str(),
        webp ? "\"extensionsUsed\":[\"EXT_texture_webp\"],\"extensionsRequired\":[\"EXT_texture_webp\"]," : "");
    appendf(json,
        "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"NORMAL\":1,\"TEXCOORD_0\":2%s},\"indices\":%d,\"material\":0,\"mode\":4}]}],"
        "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorTexture\":{\"index\":0},\"metallicRoughnessTexture\":{\"index\":1},\"metallicFactor\":1.0,\"roughnessFactor\":1.0},%s\"alphaMode\":\"OPAQUE\",\"doubleSided\":%s}],",
        has_nrm ? ",\"TANGENT\":3" : "", bvIdx,
        has_nrm ? "\"normalTexture\":{\"index\":2}," : "",
        double_sided ? "true" : "false");
    if (webp)
        json += "\"textures\":[{\"sampler\":0,\"extensions\":{\"EXT_texture_webp\":{\"source\":0}}},{\"sampler\":0,\"extensions\":{\"EXT_texture_webp\":{\"source\":1}}}";
    else
        json += "\"textures\":[{\"source\":0,\"sampler\":0},{\"source\":1,\"sampler\":0}";
    if (has_nrm) json += ",{\"source\":2,\"sampler\":0}";
    json += "],";
    appendf(json, "\"images\":[{\"bufferView\":%d,\"mimeType\":\"%s\"},{\"bufferView\":%d,\"mimeType\":\"%s\"}",
            bvImg, mime, bvImg+1, mime);
    if (has_nrm) appendf(json, ",{\"bufferView\":%d,\"mimeType\":\"image/png\"}", bvImg+2);
    json += "],\"samplers\":[{}],";
    appendf(json,
        "\"accessors\":["
        "{\"bufferView\":0,\"componentType\":5126,\"count\":%lld,\"type\":\"VEC3\",\"min\":[%.9g,%.9g,%.9g],\"max\":[%.9g,%.9g,%.9g]},"
        "{\"bufferView\":1,\"componentType\":5126,\"count\":%lld,\"type\":\"VEC3\"},"
        "{\"bufferView\":2,\"componentType\":5126,\"count\":%lld,\"type\":\"VEC2\"},",
        (long long)V, mn[0],mn[1],mn[2],mx[0],mx[1],mx[2], (long long)V, (long long)V);
    if (has_nrm) appendf(json, "{\"bufferView\":3,\"componentType\":5126,\"count\":%lld,\"type\":\"VEC4\"},", (long long)V);
    appendf(json, "{\"bufferView\":%d,\"componentType\":5125,\"count\":%lld,\"type\":\"SCALAR\"}],", bvIdx, (long long)(F*3));
    appendf(json,
        "\"bufferViews\":["
        "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":%u,\"target\":34962},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},",
        posB, posB,nrmB, posB+nrmB,uvB);
    if (has_nrm) appendf(json, "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},", posB+nrmB+uvB, tanB);
    appendf(json,
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34963},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u}",
        idxOff,idxB, offB,(uint32_t)pngB.size(), offM,(uint32_t)pngM.size());
    if (has_nrm) appendf(json, ",{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u}", offN,(uint32_t)pngN.size());
    appendf(json, "],\"buffers\":[{\"byteLength\":%u}]}", (uint32_t)bin.size());
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
