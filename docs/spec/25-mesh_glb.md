# mesh_glb


# mesh_glb — Geometry-only GLB export spec (MVP)

## 0. Scope / what this component does

Input: a generated triangle mesh
- `vertices` : float32 `[V, 3]` in TRELLIS world space
- `faces`    : int32   `[F, 3]` (zero-based vertex indices, CCW winding in TRELLIS space)

Output: one self-contained binary `.glb` file (glTF 2.0) holding a single mesh
with one primitive (POSITION + indexed triangles). No materials, no UVs, no
normals, no textures.

The Python reference (`o-voxel/o_voxel/postprocess.py::to_glb`) does a LOT more
(hole filling, simplification, remeshing, UV unwrap, texture baking, PBR
material), but ALL of that is texture/topology cleanup that we explicitly DEFER.
The only parts of `to_glb` that are load-bearing for geometry are:
1. The AABB / voxel-size convention (constants).
2. The final coordinate-system conversion (axis swap + sign flip + UV V-flip).
3. The actual glTF/GLB serialization (done by trimesh in Python; we reimplement).

## 1. Coordinate convention — CONFIRMED FROM CODE

From `postprocess.py` lines 306-323, the FINAL step before handing vertices to
the GLB serializer is:

```python
# vertices_np: (V,3) in TRELLIS space
vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
normals_np[:, 1],  normals_np[:, 2]  = normals_np[:, 2],  -normals_np[:, 1]
uvs_np[:, 1] = 1 - uvs_np[:, 1]   # UV V-flip — texture-only, SKIP for MVP
```

IMPORTANT NUMPY-TUPLE-ASSIGNMENT SEMANTICS. The line
`a[:,1], a[:,2] = a[:,2], -a[:,1]` evaluates the RHS tuple FIRST (snapshotting
`old_y = a[:,1]`, `old_z = a[:,2]`), then assigns. So the mapping is:

```
new_x =  old_x
new_y =  old_z
new_z = -old_y
```

i.e. (x, y, z)_trellis -> (x, z, -y)_glb. This is the standard +Z-up (TRELLIS)
-> +Y-up (glTF) rotation: -90° about the X axis. Per-vertex C++:

```c
for (v in vertices) {
    float ox = v.x, oy = v.y, oz = v.z;
    v.x =  ox;
    v.y =  oz;
    v.z = -oy;
}
```

Faces / winding: NOT touched by the Python code, indices are emitted unchanged.
This rotation is a proper rotation (det = +1), so triangle winding (front/back
face orientation) is preserved. Emit faces as-is. (`doubleSided=True` is set on
the Python material anyway, so winding is not even visually critical — but with a
proper rotation it is correct regardless.)

The UV V-flip is texture-only and is SKIPPED (no UVs in MVP).

## 2. AABB / scale convention — CONFIRMED FROM CODE

From `example.py` lines 33-46, the call site fixes:
```
aabb       = [[-0.5,-0.5,-0.5], [0.5,0.5,0.5]]   # unit cube centered at origin
voxel_size = mesh.voxel_size                       # from the SS/decoder stage
```
`postprocess.to_glb` derives `grid_size = round((aabb[1]-aabb[0]) / voxel_size)`
(line 78). These are ONLY used for texture-volume sampling (`grid_sample_3d`) and
remeshing center/scale — NONE feed into vertex positions for the geometry path.

The vertices arriving at `to_glb` are ALREADY in this `[-0.5, 0.5]^3` world cube
(mesh extraction places them there). So for the geometry MVP we do NOT rescale,
re-center, or use aabb/voxel_size at all — vertices pass straight through the
coordinate rotation above. Record `[-0.5,0.5]^3` only for an optional sanity
clamp/assert.

`MeshWithVoxel.__init__` (mesh/base.py 190-208) stores `origin`, `voxel_size`,
`coords`, `attrs`, `voxel_shape`, `layout` — these are the volume/attribute
fields used only by texture baking; ignore for geometry.

## 3. GLB container layout (glTF 2.0 binary) — exact bytes

A `.glb` is: a 12-byte header, then a sequence of chunks. We emit exactly two
chunks: JSON then BIN. All multi-byte integers are little-endian uint32.

### 3.1 Header (12 bytes)
| off | type   | value                                   |
|-----|--------|-----------------------------------------|
| 0   | u32 LE | magic   = 0x46546C67  ("glTF" ASCII)    |
| 4   | u32 LE | version = 2                             |
| 8   | u32 LE | length  = total file size in bytes      |

`length` = 12 (header) + 8 + jsonChunkLen + 8 + binChunkLen, where the two `8`s
are the per-chunk (length,type) prefixes and jsonChunkLen/binChunkLen are the
PADDED chunk payload lengths (see 3.4).

### 3.2 JSON chunk
| off            | type   | value                                |
|----------------|--------|--------------------------------------|
| 0              | u32 LE | chunkLength = padded JSON byte len   |
| 4              | u32 LE | chunkType   = 0x4E4F534A ("JSON")    |
| 8              | bytes  | UTF-8 JSON, padded with 0x20 (space) |

### 3.3 BIN chunk
| off            | type   | value                                |
|----------------|--------|--------------------------------------|
| 0              | u32 LE | chunkLength = padded BIN byte len    |
| 4              | u32 LE | chunkType   = 0x004E4942 ("BIN\0")   |
| 8              | bytes  | raw buffer data, padded with 0x00    |

### 3.4 Alignment rules (REQUIRED by spec)
- Each chunk's payload length MUST be a multiple of 4.
- JSON chunk: pad with trailing space chars `0x20` to next multiple of 4.
- BIN chunk: pad with trailing zero bytes `0x00` to next multiple of 4.
- The BIN chunk MUST start at a 4-byte boundary — guaranteed because header(12)
  + 8 + paddedJsonLen is always a multiple of 4.
- Inside the buffer, each accessor's data must begin at an offset that is a
  multiple of its component size. We place indices first then positions, or
  vice-versa, with explicit byteOffset padding (see 4.2). Simplest correct
  layout: positions (float, 4-byte aligned) at offset 0, then indices
  (uint32, 4-byte aligned) immediately after — both are 4-byte aligned naturally,
  no internal padding needed.

## 4. BIN buffer contents

Two contiguous regions in ONE buffer (buffer 0). Choose this order (both 4-byte
natural alignment, zero internal padding):

### 4.1 Region A — POSITIONS
- bytes: `V * 3 * 4` (float32, little-endian)
- layout: `x0 y0 z0 x1 y1 z1 ...` AFTER the coordinate rotation in §1.
- byteOffset in buffer: 0
- accessor: componentType 5126 (FLOAT), type "VEC3", count V
- MUST include `min` and `max`: per-component element-wise min/max over the
  POST-rotation positions (3-float arrays). glTF requires min/max on POSITION.

### 4.2 Region B — INDICES
- bytes: `F * 3 * 4` (uint32, little-endian)  [use 5125 UNSIGNED_INT]
- layout: flattened faces `i0 i1 i2 i3 i4 i5 ...`
- byteOffset in buffer: `V * 3 * 4` (already 4-byte aligned)
- accessor: componentType 5125 (UNSIGNED_INT), type "SCALAR", count F*3

(Optional micro-optimization: if `max_index < 65536` use 5123 UNSIGNED_SHORT and
2-byte indices; then pad the indices region to a multiple of 4 before the buffer
end. For MVP simplicity, always use UNSIGNED_INT 5125 to avoid the alignment
special-case.)

Total buffer length (unpadded) = `V*12 + F*12`. Pad to multiple of 4 (already is,
since both terms are multiples of 4) for the BIN chunk.

## 5. JSON document (exact structure)

Minimal valid glTF 2.0 with one scene, one node, one mesh, one primitive, two
accessors, two bufferViews, one buffer, one asset block. Use `byteLength` of the
UNPADDED buffer in `buffers[0].byteLength` (glTF allows the BIN chunk to be
padded beyond byteLength; buffers[0] has NO `uri` since data is in the BIN chunk).

```json
{
  "asset": { "version": "2.0", "generator": "trellis.cpp" },
  "scene": 0,
  "scenes": [ { "nodes": [0] } ],
  "nodes":  [ { "mesh": 0 } ],
  "meshes": [ {
    "primitives": [ {
      "attributes": { "POSITION": 0 },
      "indices": 1,
      "mode": 4
    } ]
  } ],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": V,
      "type": "VEC3", "min": [minx,miny,minz], "max": [maxx,maxy,maxz] },
    { "bufferView": 1, "componentType": 5125, "count": F3,
      "type": "SCALAR" }
  ],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0,        "byteLength": V*12, "target": 34962 },
    { "buffer": 0, "byteOffset": V*12,     "byteLength": F*12, "target": 34963 }
  ],
  "buffers": [ { "byteLength": V*12 + F*12 } ]
}
```
- `mode: 4` = TRIANGLES.
- `target: 34962` = ARRAY_BUFFER (vertex attrs); `34963` = ELEMENT_ARRAY_BUFFER
  (indices). Optional but conventional; safe to include.
- `F3 = F*3`.
- Do NOT emit a trailing newline inside the JSON payload; pad with spaces.
- Emit numbers with enough precision: positions are float32, so print min/max
  with e.g. 9 significant digits (`%.9g`) to round-trip.

## 6. Skip list (explicitly NOT in MVP)
- `mesh.fill_holes`, `mesh.simplify`, `mesh.remove_duplicate_faces`,
  `repair_non_manifold_edges`, `remove_small_connected_components`,
  `unify_face_orientations` — all `cumesh` CUDA ops.
- `remesh_narrow_band_dc` (Dual Contouring remesh) — cumesh CUDA.
- `mesh.uv_unwrap`, vertex normals, `out_vmaps` — UV/normal generation.
- nvdiffrast rasterization + `dr.interpolate` + `bvh.unsigned_distance` —
  texture baking.
- `grid_sample_3d` attribute sampling, `cv2.inpaint`, PBR material, base_color/
  metallic/roughness/alpha textures — all texture pipeline.
- UV V-flip (`uvs_np[:,1] = 1 - uvs_np[:,1]`) — no UVs.
- normals axis swap — no normals (let viewers compute flat normals).
- `example.py` `mesh.simplify(16777216)` (nvdiffrast vertex-count cap) — only
  needed if downstream rasterization; not for export.

The geometry MVP keeps ONLY: coordinate rotation (§1) + GLB serialization (§3-5).


## Weight key map


NONE. mesh_glb is a pure serialization/export component with ZERO learned
parameters. It consumes runtime tensors produced by upstream stages, not
safetensors/gguf weights:
- vertices [V,3] float32 — output of the SS conv3d decoder + mesh extraction
  (Stage ① / cumesh marching, already produced as active-voxel-derived geometry).
- faces [F,3] int32 — same source.

No entry in any of /devel/alt/trellis.cpp/docs/spec/keys/*.keys.txt corresponds
to this component (verified: the key dumps are all DiT / encoder / decoder /
conditioner weights: ss_dec_conv3d, shape_enc/dec, slat_flow, ss_flow, dinov3,
birefnet — none are mesh-export params). Nothing to map.


## GGML plan


GGML is NOT needed for this component — it is plain CPU C++ byte I/O. No tensors,
no compute graph, no kernels.

Recommended implementation (single self-contained writer, no external deps):

File: src/mesh_glb.cpp + include/mesh_glb.h
Signature:
  bool write_glb(const char* path,
                 const float* verts, int64_t V,    // [V*3], TRELLIS space
                 const int32_t* faces, int64_t F); // [F*3], zero-based

Steps:
1. Coordinate rotation in a local std::vector<float> pos(V*3):
     for i in [0,V): ox=verts[3i],oy=verts[3i+1],oz=verts[3i+2];
       pos[3i]=ox; pos[3i+1]=oz; pos[3i+2]=-oy;
   Track per-component min/max during this loop (init to +/-FLT_MAX).
2. Build index buffer std::vector<uint32_t> idx(F*3) = copy of faces (cast int32
   -> uint32; assert all >=0 and < V).
3. Build BIN buffer (std::vector<uint8_t> bin): memcpy pos (V*12 bytes) then idx
   (F*12 bytes). On a little-endian host (x86_64, our target) raw memcpy is
   correct for the LE glTF requirement — no byte swapping. (Add a static_assert
   or runtime endianness note; all CI/target machines here are x86_64 LE.)
4. Build JSON string per §5 via snprintf/std::string concat; substitute V, F*3,
   byteOffsets, byteLength, and the 6 min/max floats (%.9g).
5. Pad JSON payload with ' ' (0x20) to multiple of 4. Pad BIN payload with 0x00
   to multiple of 4 (already aligned here, but do it defensively).
6. Compute totalLen = 12 + 8 + jsonPadLen + 8 + binPadLen.
7. fwrite in order: header(magic,version,totalLen) -> jsonChunk(len,type,
   payload) -> binChunk(len,type,payload). Use a small le_u32 writer that emits
   4 bytes little-endian (portable even if host were BE).

Memory: O(V*12 + F*12) transient host buffers; negligible. No GPU.

Why no ggml: the existing helpers (Linear, conv_3d, RoPE, SDPA in src/dit.cpp /
ss_decoder.cpp) are irrelevant — there is no math here beyond a per-vertex axis
permutation, which is cheaper inline than a ggml graph.

Endianness helper (the only subtlety):
  static void w_u32(std::vector<uint8_t>& o, uint32_t v){
    o.push_back(v&0xff); o.push_back((v>>8)&0xff);
    o.push_back((v>>16)&0xff); o.push_back((v>>24)&0xff); }
Float payload memcpy is fine on LE host; if portability to BE is ever required,
serialize floats via their bit pattern with the same LE order.


## Reuse


REUSE: essentially none of the DiT/conv/flow helpers apply — this is I/O, not
tensor compute. Do NOT pull in ggml for it.

Patterns worth mirroring from the existing tree:
- include/npy.h already establishes the project's binary-file read/write idiom
  (raw fwrite of little-endian POD, header string assembly). mesh_glb.cpp should
  follow the same plain-C++ style (std::ofstream/fwrite, std::vector<uint8_t>
  scratch). Read /devel/alt/trellis.cpp/include/npy.h before writing for the
  house style of LE serialization.
- The upstream Python write_ply (mesh_utils.py:120 write_ply, and write_pbr_ply)
  is a useful reference for the *PLY* alternative if a quick debug dump is wanted
  before GLB works — it is dependency-light (binary_little_endian, struct.pack
  '<fff' verts, '<B3i' tri faces). A C++ binary-PLY writer is ~30 lines and can
  serve as an intermediate validation artifact (load in MeshLab) independent of
  the coordinate rotation. PLY in TRELLIS uses NO axis swap (it stores raw
  vertices); only the GLB path applies the (x,z,-y) rotation.

GENUINELY NEW (must be written from scratch):
- The GLB chunk/header serializer (no equivalent in repo).
- The minimal glTF2 JSON emitter.
- The (x,y,z)->(x,z,-y) vertex rotation + POSITION min/max accumulation.

Integration point: call write_glb() at the end of the Stage-② mesh pipeline,
taking the vertices/faces tensors that the (future) mesh-extraction step produces
from SS decoder active voxels. For now it can be exercised by a standalone test
(mirroring src/test_ss_dec.cpp style) that loads a vertices.npy + faces.npy via
include/npy.h and writes out.glb.


## Open questions


1. Mesh extraction source/space: confirm that the vertices handed to export are
   already in the [-0.5,0.5]^3 world cube (as example.py's aabb implies) and NOT
   in voxel-index space. The Python to_glb does no rescale on the geometry path,
   so this is almost certainly true, but it depends on the (not-yet-ported)
   marching/dual-contouring extraction stage. Action: when that stage lands, dump
   one vertices tensor and assert bounds within [-0.5,0.5] (+ small margin).

2. Winding/orientation: Python sets doubleSided=True so it never validated single-
   sided winding. With a proper rotation (det=+1) the original winding is
   preserved, so a correct extractor's CCW triangles stay CCW in glTF (+Y-up).
   If the upstream extractor emits CW or inconsistent winding, single-sided
   viewers may show inverted faces. Action: visually verify in a glTF viewer; if
   needed add an optional winding flip or emit a doubleSided-equivalent (glTF has
   no geometry-level doubleSided without a material — would require a minimal
   material with doubleSided:true; defer).

3. Index width: do we ever exceed 2^32-1 indices? V*3 for million-vert meshes is
   ~3e6 << 2^32, so UNSIGNED_INT is always safe; UNSIGNED_SHORT optimization is
   optional. No action needed unless file-size matters.

4. Float precision for min/max: %.9g should round-trip float32. Confirm chosen
   glTF validator (if any) is satisfied; trivial to widen.

5. Does any downstream consumer require normals/UVs even for a "geometry MVP"
   (e.g., the validation viewer renders black without normals)? Most viewers
   compute flat normals from faces. Action: if rendering looks unlit, add a
   POSTPROCESS step generating per-face flat normals as a separate accessor —
   small, optional follow-up.

