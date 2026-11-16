// Minimal stand-in for <torch/extension.h>, just enough to compile o-voxel's
// src/convert/flexible_dual_grid.cpp without libtorch (tools/ref_voxelize, test oracle only).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <array>
#include <memory>
#include <tuple>
#include <vector>

namespace torch {
enum Dtype { kInt32, kFloat32, kBool };
inline size_t dtype_size(Dtype d) { return d == kBool ? 1 : 4; }

class Tensor {
public:
    Tensor() = default;
    Tensor(void* p, std::vector<int64_t> shape, Dtype dt, bool own)
        : shape_(std::move(shape)), dt_(dt) {
        size_t n = dtype_size(dt);
        for (int64_t s : shape_) n *= (size_t)s;
        if (own) { buf_ = std::make_shared<std::vector<char>>(n); if (n) std::memcpy(buf_->data(), p, n); data_ = buf_->data(); }
        else data_ = p;
    }
    template <typename T> T* data_ptr() const { return reinterpret_cast<T*>(data_); }
    int64_t size(int i) const { return shape_[(size_t)i]; }
    Tensor clone() const { return Tensor(data_, shape_, dt_, true); }
    size_t nbytes() const { size_t n = dtype_size(dt_); for (int64_t s : shape_) n *= (size_t)s; return n; }
private:
    std::vector<int64_t> shape_;
    Dtype dt_ = kFloat32;
    void* data_ = nullptr;
    std::shared_ptr<std::vector<char>> buf_;
};

inline Tensor from_blob(void* p, std::initializer_list<int64_t> shape, Dtype dt) {
    return Tensor(p, std::vector<int64_t>(shape), dt, false);
}
}  // namespace torch
