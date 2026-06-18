// Minimal .npy reader/writer for float32 C-order arrays (validation only).
#pragma once
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <numeric>

namespace npy {

struct Array {
    std::vector<int64_t> shape;
    std::vector<float>   data;
    int64_t numel() const {
        return std::accumulate(shape.begin(), shape.end(), (int64_t)1, std::multiplies<int64_t>());
    }
};

inline Array load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("npy: cannot open " + path);
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("npy: bad magic " + path);
    uint16_t hlen;
    if (fread(&hlen, 2, 1, f) != 1) throw std::runtime_error("npy: header len");
    std::string hdr(hlen, '\0');
    if (fread(hdr.data(), 1, hlen, f) != hlen) throw std::runtime_error("npy: header");
    if (hdr.find("'<f4'") == std::string::npos && hdr.find("\"<f4\"") == std::string::npos)
        throw std::runtime_error("npy: only <f4 supported: " + path + " hdr=" + hdr);
    if (hdr.find("'fortran_order': True") != std::string::npos)
        throw std::runtime_error("npy: fortran order unsupported");
    // parse shape tuple
    Array a;
    size_t p = hdr.find("'shape':");
    p = hdr.find('(', p);
    size_t q = hdr.find(')', p);
    std::string s = hdr.substr(p + 1, q - p - 1);
    for (size_t i = 0; i < s.size();) {
        if (isdigit(s[i])) {
            int64_t v = 0; while (i < s.size() && isdigit(s[i])) v = v * 10 + (s[i++] - '0');
            a.shape.push_back(v);
        } else ++i;
    }
    if (a.shape.empty()) a.shape.push_back(1);
    a.data.resize(a.numel());
    if ((int64_t)fread(a.data.data(), sizeof(float), a.numel(), f) != a.numel())
        throw std::runtime_error("npy: short data " + path);
    fclose(f);
    return a;
}

inline void save(const std::string& path, const float* data, const std::vector<int64_t>& shape) {
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); ++i) sh += std::to_string(shape[i]) + (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
    sh += ")";
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t total = 10 + hdr.size() + 1;
    size_t pad = (64 - (total % 64)) % 64;
    hdr.append(pad, ' '); hdr += '\n';
    uint16_t hlen = (uint16_t)hdr.size();
    FILE* f = fopen(path.c_str(), "wb");
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite(&hlen, 2, 1, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    int64_t n = std::accumulate(shape.begin(), shape.end(), (int64_t)1, std::multiplies<int64_t>());
    fwrite(data, sizeof(float), n, f);
    fclose(f);
}

} // namespace npy
