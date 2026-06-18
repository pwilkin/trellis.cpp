// Debug: my SS sampler+decode on the REAL goblin cond + reference noise.
//   trellis-test-ss-full <gguf_dir> <ref_root> [gpu]
#include "trellis_model.h"
#include "flow_runner.h"
#include "ss_decoder.h"
#include "npy.h"
#include <cmath>
#include <cstdio>
#include <vector>

using std::vector;
int main(int argc, char** argv) {
    const std::string G = argv[1], R = argv[2];
    const int gpu = argc > 3 ? atoi(argv[3]) : 1;
    npy::Array noise = npy::load(R + "/ss_full/noise.npy");  // [1,8,16,16,16] torch c*4096+sp
    npy::Array cond = npy::load(R + "/dinov3/cond.npy");      // [1,1029,1024]
    npy::Array zref = npy::load(R + "/ss_full/z.npy");        // [1,8,16,16,16]
    const int Lc = (int)cond.shape[1];
    vector<float> neg(cond.data.size(), 0.0f);

    // noise torch [8,4096] (c*4096+sp) -> my [8,L] ne0=8 (c + 8*sp)
    vector<float> nz(8*4096);
    for (int c = 0; c < 8; ++c) for (int sp = 0; sp < 4096; ++sp) nz[c + 8*sp] = noise.data[(size_t)c*4096 + sp];

    trellis::Model m = trellis::Model::load(G + "/ss_flow.gguf", gpu);
    trellis::DiTParams p; p.in_ch = 8; p.out_ch = 8; p.d_cond = 1024;
    if (getenv("TRELLIS_F32W")) p.cast_f32 = true;
    trellis::DitRunner* run = trellis::make_dense_runner(m, p, 16, Lc);
    trellis::FlowFwd fwd = [&](const vector<float>& x, float ts, const float* c){ return run->forward(x, ts, c); };
    trellis::SamplerParams sp; sp.steps=12; sp.guidance_strength=7.5f; sp.guidance_rescale=0.7f; sp.gi0=0.6f; sp.gi1=1.0f; sp.rescale_t=5.0f;
    if (getenv("GS")) sp.guidance_strength = atof(getenv("GS"));
    if (getenv("GR")) sp.guidance_rescale = atof(getenv("GR"));
    vector<float> z = trellis::sample_flow(fwd, nz, cond.data.data(), neg.data(), sp);  // [8,L]
    delete run; m.free();

    // compare to ref z (transpose ref c*4096+sp -> [8,L] c+8*sp)
    double maxd=0, gmax=0; for (int c=0;c<8;++c) for (int s=0;s<4096;++s){ double d=std::fabs((double)z[c+8*s]-zref.data[(size_t)c*4096+s]); maxd=std::max(maxd,d); gmax=std::max(gmax,std::fabs((double)zref.data[(size_t)c*4096+s])); }
    printf("z vs ref: rel=%.4e\n", maxd/gmax);
    { double s=0,s2=0; for(float v:z){s+=v;s2+=(double)v*v;} printf("my z std=%.4f\n", std::sqrt(s2/z.size()-(s/z.size())*(s/z.size()))); }

    // decode
    vector<float> zdec(8*4096);
    for (int c=0;c<8;++c) for (int s=0;s<4096;++s) zdec[(size_t)c*4096+s]=z[c+8*s];
    trellis::Model d = trellis::Model::load(G + "/ss_dec.gguf", gpu);
    vector<float> logits = trellis::ss_decode(d, zdec); d.free();
    int pos=0; double mn=1e9,mx=-1e9; for(float v:logits){if(v>0)pos++;mn=std::min(mn,(double)v);mx=std::max(mx,(double)v);}
    printf("my decode: logits %.2f/%.2f  voxels@64=%d\n", mn, mx, pos);
    // also decode the REFERENCE z with my decoder (isolates decode vs sampler)
    vector<float> zr(8*4096); for (size_t i=0;i<zr.size();++i) zr[i]=zref.data[i];
    trellis::Model d2 = trellis::Model::load(G + "/ss_dec.gguf", gpu);
    vector<float> lr = trellis::ss_decode(d2, zr); d2.free();
    int pr=0; for(float v:lr) if(v>0)pr++;
    printf("REF z my-decode: voxels@64=%d (torch said 5406)\n", pr);
    return 0;
}
