// Minimal GLB (binary glTF 2.0) inspection: triangle count and embedded textures.
//
// Studio uses this to:
//   - show each result's real face count in the viewer caption;
//   - notice when the server ignored a requested setting (a trellis-server
//     that predates `faces`/`atlas` drops unknown form fields silently);
//   - pull the textures out as standalone image files for the modeler
//     handoff package (handoff.ts).
//
// Only the 12-byte header, the JSON chunk and the byte ranges of the images
// are read; vertex data is never copied.
//
// Layout (glTF 2.0 spec, "GLB File Format Specification"):
//   header:  u32 magic "glTF" | u32 version (2) | u32 total length
//   chunk 0: u32 length | u32 type "JSON" | UTF-8 JSON
//   chunk 1: u32 length | u32 type "BIN\0" | binary buffer 0

const MAGIC_GLTF = 0x46546c67; // "glTF" little-endian
const CHUNK_JSON = 0x4e4f534a; // "JSON"
const CHUNK_BIN = 0x004e4942; // "BIN\0"
const MODE_TRIANGLES = 4; // glTF default primitive mode

interface TextureRef {
  index: number;
}
interface GltfJson {
  accessors?: { count: number }[];
  bufferViews?: { buffer: number; byteOffset?: number; byteLength: number }[];
  images?: { bufferView?: number; mimeType?: string; uri?: string }[];
  textures?: { source?: number; extensions?: { EXT_texture_webp?: { source: number } } }[];
  materials?: {
    pbrMetallicRoughness?: { baseColorTexture?: TextureRef; metallicRoughnessTexture?: TextureRef };
    normalTexture?: TextureRef;
  }[];
  meshes?: { primitives: { indices?: number; mode?: number; attributes: Record<string, number> }[] }[];
}

/** What a texture is used for; becomes part of its file name in the handoff package. */
export type TextureRole = "basecolor" | "metallic_roughness" | "normal" | "texture";

/** One image embedded in the GLB. */
export interface GlbImage {
  role: TextureRole;
  mimeType: string; // "image/png" | "image/webp" | ...
  data: Blob;
}

export interface GlbInfo {
  faces: number;
  images: GlbImage[];
}

/** File extension for an image MIME type ("image/webp" -> "webp"). */
export function imageExt(mimeType: string): string {
  return mimeType.split("/")[1]?.replace("jpeg", "jpg") || "bin";
}

/**
 * Read the JSON chunk, and note where the BIN chunk's data starts.
 * @throws if the blob is not a GLB 2.0 file with a leading JSON chunk.
 */
async function readChunks(glb: Blob): Promise<{ json: GltfJson; binStart: number | null }> {
  const head = new DataView(await glb.slice(0, 20).arrayBuffer());
  if (head.byteLength < 20 || head.getUint32(0, true) !== MAGIC_GLTF || head.getUint32(4, true) !== 2) {
    throw new Error("not a glTF 2.0 binary");
  }
  const jsonLength = head.getUint32(12, true);
  if (head.getUint32(16, true) !== CHUNK_JSON) throw new Error("GLB has no leading JSON chunk");
  const json = JSON.parse(await glb.slice(20, 20 + jsonLength).text()) as GltfJson;

  // The BIN chunk, if present, follows the (4-byte padded) JSON chunk.
  const binHeaderAt = 20 + jsonLength;
  const binHead = new DataView(await glb.slice(binHeaderAt, binHeaderAt + 8).arrayBuffer());
  const hasBin = binHead.byteLength === 8 && binHead.getUint32(4, true) === CHUNK_BIN;
  return { json, binStart: hasBin ? binHeaderAt + 8 : null };
}

/** Triangles across all meshes: indices/3 for indexed primitives, POSITION/3 otherwise. */
function countFaces(json: GltfJson): number {
  const accessors = json.accessors ?? [];
  let faces = 0;
  for (const mesh of json.meshes ?? []) {
    for (const prim of mesh.primitives) {
      if ((prim.mode ?? MODE_TRIANGLES) !== MODE_TRIANGLES) continue;
      const accessor = prim.indices ?? prim.attributes.POSITION;
      faces += Math.floor((accessors[accessor]?.count ?? 0) / 3);
    }
  }
  return faces;
}

/** Map image index -> role, from how the materials use each texture. */
function imageRoles(json: GltfJson): Map<number, TextureRole> {
  const roles = new Map<number, TextureRole>();
  const imageOf = (ref?: TextureRef) => {
    const tex = ref ? json.textures?.[ref.index] : undefined;
    // EXT_texture_webp puts the image index in the extension (trellis writes no fallback).
    return tex?.extensions?.EXT_texture_webp?.source ?? tex?.source;
  };
  for (const m of json.materials ?? []) {
    const set = (ref: TextureRef | undefined, role: TextureRole) => {
      const img = imageOf(ref);
      if (img !== undefined && !roles.has(img)) roles.set(img, role);
    };
    set(m.pbrMetallicRoughness?.baseColorTexture, "basecolor");
    set(m.pbrMetallicRoughness?.metallicRoughnessTexture, "metallic_roughness");
    set(m.normalTexture, "normal");
  }
  return roles;
}

/** Number of triangles across all meshes in a GLB (reads only the header + JSON). */
export async function glbFaceCount(glb: Blob): Promise<number> {
  return countFaces((await readChunks(glb)).json);
}

/**
 * Face count plus every image embedded in the GLB's binary buffer, each
 * tagged with its material role. Images stored as external URIs (never
 * produced by trellis) are skipped.
 */
export async function inspectGlb(glb: Blob): Promise<GlbInfo> {
  const { json, binStart } = await readChunks(glb);
  const roles = imageRoles(json);
  const images: GlbImage[] = [];
  (json.images ?? []).forEach((img, i) => {
    const view = img.bufferView !== undefined ? json.bufferViews?.[img.bufferView] : undefined;
    if (!view || view.buffer !== 0 || binStart === null) return;
    const start = binStart + (view.byteOffset ?? 0);
    const mimeType = img.mimeType ?? "application/octet-stream";
    images.push({
      role: roles.get(i) ?? "texture",
      mimeType,
      data: glb.slice(start, start + view.byteLength, mimeType),
    });
  });
  return { faces: countFaces(json), images };
}
