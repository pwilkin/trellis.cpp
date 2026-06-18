#include "mesh_glb.h"
#include <cstdio>
#include <cstring>
#include <cfloat>
#include <string>
#include <vector>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace trellis {

static void png_collect(void* ctx, void* data, int size) {
    auto* v = (std::vector<uint8_t>*)ctx;
    v->insert(v->end(), (uint8_t*)data, (uint8_t*)data + size);
}

static void w_u32(std::vector<uint8_t>& o, uint32_t v) {
    o.push_back(v & 0xff); o.push_back((v >> 8) & 0xff);
    o.push_back((v >> 16) & 0xff); o.push_back((v >> 24) & 0xff);
}

bool write_glb(const char* path, const float* verts, int64_t V, const int32_t* faces, int64_t F,
               const float* colors) {
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

    char buf[2048];
    std::snprintf(buf, sizeof(buf),
        "{\"asset\":{\"version\":\"2.0\",\"generator\":\"trellis.cpp\"},"
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
        attr, (long long)V, mn[0], mn[1], mn[2], mx[0], mx[1], mx[2],
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
                        const unsigned char* base_rgba, const unsigned char* mr_rgba, int T) {
    // rotate positions (x,z,-y) + min/max
    std::vector<float> pos((size_t)V*3);
    float mn[3]={FLT_MAX,FLT_MAX,FLT_MAX}, mx[3]={-FLT_MAX,-FLT_MAX,-FLT_MAX};
    for (int64_t i=0;i<V;++i){ float ox=verts[3*i],oy=verts[3*i+1],oz=verts[3*i+2]; float p[3]={ox,oz,-oy};
        for(int c=0;c<3;++c){pos[3*i+c]=p[c]; mn[c]=std::min(mn[c],p[c]); mx[c]=std::max(mx[c],p[c]);} }
    if (V==0){mn[0]=mn[1]=mn[2]=0;mx[0]=mx[1]=mx[2]=0;}

    // encode textures to PNG
    std::vector<uint8_t> pngB, pngM;
    stbi_write_png_to_func(png_collect, &pngB, T, T, 4, base_rgba, T*4);
    stbi_write_png_to_func(png_collect, &pngM, T, T, 4, mr_rgba, T*4);

    auto pad4=[](std::vector<uint8_t>&b){ while(b.size()%4) b.push_back(0); };
    const uint32_t posB=(uint32_t)(V*12), uvB=(uint32_t)(V*8), idxB=(uint32_t)(F*12);
    std::vector<uint8_t> bin(posB+uvB+idxB);
    std::memcpy(bin.data(), pos.data(), posB);
    std::memcpy(bin.data()+posB, uv, uvB);
    for (int64_t i=0;i<F*3;++i){ uint32_t v=(uint32_t)faces[i]; std::memcpy(bin.data()+posB+uvB+i*4,&v,4); }
    pad4(bin); const uint32_t off3=(uint32_t)bin.size();
    bin.insert(bin.end(), pngB.begin(), pngB.end()); pad4(bin); const uint32_t off4=(uint32_t)bin.size();
    bin.insert(bin.end(), pngM.begin(), pngM.end()); pad4(bin);

    char buf[2400];
    std::snprintf(buf,sizeof(buf),
        "{\"asset\":{\"version\":\"2.0\",\"generator\":\"trellis.cpp\"},"
        "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],"
        "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"TEXCOORD_0\":1},\"indices\":2,\"material\":0,\"mode\":4}]}],"
        "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorTexture\":{\"index\":0},\"metallicRoughnessTexture\":{\"index\":1},\"metallicFactor\":1.0,\"roughnessFactor\":1.0},\"doubleSided\":true}],"
        "\"textures\":[{\"source\":0,\"sampler\":0},{\"source\":1,\"sampler\":0}],"
        "\"images\":[{\"bufferView\":3,\"mimeType\":\"image/png\"},{\"bufferView\":4,\"mimeType\":\"image/png\"}],"
        "\"samplers\":[{}],"
        "\"accessors\":["
        "{\"bufferView\":0,\"componentType\":5126,\"count\":%lld,\"type\":\"VEC3\",\"min\":[%.9g,%.9g,%.9g],\"max\":[%.9g,%.9g,%.9g]},"
        "{\"bufferView\":1,\"componentType\":5126,\"count\":%lld,\"type\":\"VEC2\"},"
        "{\"bufferView\":2,\"componentType\":5125,\"count\":%lld,\"type\":\"SCALAR\"}],"
        "\"bufferViews\":["
        "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":%u,\"target\":34962},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34963},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u}],"
        "\"buffers\":[{\"byteLength\":%u}]}",
        (long long)V, mn[0],mn[1],mn[2],mx[0],mx[1],mx[2],
        (long long)V, (long long)(F*3),
        posB, posB,uvB, posB+uvB,idxB,
        off3,(uint32_t)pngB.size(), off4,(uint32_t)pngM.size(), (uint32_t)bin.size());
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
