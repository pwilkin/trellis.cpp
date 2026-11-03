// Modeler handoff package: everything a 3D modeler needs to rebuild a
// generated asset properly, in one folder.
//
//   <output>/handoff/<asset>_<id>/
//     <asset>.glb                       the generated mesh, untouched
//     <asset>_source.<ext>              the concept image it was generated from
//     <asset>_basecolor.<ext>           textures, extracted from the GLB so they
//     <asset>_metallic_roughness.<ext>    open in any image editor
//     <asset>_views_textured.png        turnaround sheet (0/90/180/270°)
//     <asset>_views_clay.png            same views, untextured: the form alone
//     notes.txt                         settings, reproducibility, caveats
//
// The GLB is reference, not a production asset: trellis output is dense,
// triangulated and auto-UV'd. notes.txt says so, along with what the modeler
// can't tell from the files alone (scale, axes, which sides were guessed).
//
// This module only assembles the files (as Blobs). Writing them to disk is
// tauri.ts (writeHandoffFolder), so the assembly stays testable in a browser.

import { imageExt, inspectGlb, type GlbInfo } from "./glb";
import { bgLabel, budgetOf, fovOf, presetName } from "./options";
import { formatFaces, type GenRecord } from "./types";
import type { Viewer } from "./viewer";

export interface HandoffFile {
  name: string;
  data: Blob;
}

export interface HandoffPackage {
  /** Folder name: "<asset>_<record id>", unique per generation. */
  folder: string;
  files: HandoffFile[];
  /** True when the textures are WebP, which some DCC tools can't open (see notes). */
  webpTextures: boolean;
}

/** "Iron Chest (final).png" -> "Iron_Chest_final": safe in file names on every OS. */
export function assetBaseName(inputName: string): string {
  const stem = inputName.replace(/\.[^.]+$/, "");
  const safe = stem.replace(/[^\p{L}\p{N}_-]+/gu, "_").replace(/^_+|_+$/g, "");
  return safe || "asset";
}

/**
 * Build the package for one gallery record.
 *
 * @param rec    The generation to export (its GLB, input image and settings).
 * @param viewer The live viewer, which must be showing `rec` (the turnaround
 *               sheets are rendered from it) and be visible on screen.
 */
export async function buildHandoff(rec: GenRecord, viewer: Viewer): Promise<HandoffPackage> {
  const base = assetBaseName(rec.name);
  const info = await inspectGlb(rec.glb);
  const files: HandoffFile[] = [
    { name: `${base}.glb`, data: rec.glb },
    { name: `${base}_source.${imageExt(rec.input.type || "image/png")}`, data: rec.input },
  ];

  // Textures, named by role; a role used twice (not produced by trellis) gets a suffix.
  const seen = new Map<string, number>();
  for (const img of info.images) {
    const n = (seen.get(img.role) ?? 0) + 1;
    seen.set(img.role, n);
    const suffix = n > 1 ? `_${n}` : "";
    files.push({ name: `${base}_${img.role}${suffix}.${imageExt(img.mimeType)}`, data: img.data });
  }

  // Turnaround sheets, rendered from the live viewer.
  files.push({ name: `${base}_views_textured.png`, data: await viewer.captureSheet({ clay: false }) });
  files.push({ name: `${base}_views_clay.png`, data: await viewer.captureSheet({ clay: true }) });

  const webpTextures = info.images.some((i) => i.mimeType === "image/webp");
  const notes = await handoffNotes(rec, info, files, webpTextures);
  files.push({ name: "notes.txt", data: new Blob([notes], { type: "text/plain" }) });

  return { folder: `${base}_${rec.id}`, files, webpTextures };
}

/** What each file in the package is, for the Files section of notes.txt. */
function describeFile(name: string): string {
  if (name.endsWith(".glb")) return "generated mesh with textures (glTF 2.0 binary)";
  if (name.includes("_source.")) return "the concept image the model was generated from";
  if (name.includes("_basecolor")) return "base color texture";
  if (name.includes("_metallic_roughness")) return "glTF packing: G = roughness, B = metallic";
  if (name.includes("_normal")) return "normal map";
  if (name.endsWith("_views_textured.png")) return "turnaround sheet: 0° / 90° / 180° / 270°";
  if (name.endsWith("_views_clay.png")) return "same views, untextured (form only)";
  return "texture";
}

/** The notes.txt text: settings for reproducibility, a file list, and caveats for the modeler. */
async function handoffNotes(
  rec: GenRecord,
  info: GlbInfo,
  files: HandoffFile[],
  webpTextures: boolean,
): Promise<string> {
  const p = rec.params;
  const base = info.images.find((i) => i.role === "basecolor");
  const bmp = base ? await createImageBitmap(base.data).catch(() => null) : null;
  const atlas = bmp ? `${bmp.width} x ${bmp.height}` : p.atlas ? `${p.atlas} (requested)` : "default";
  bmp?.close();
  const nameWidth = Math.max(...files.map((f) => f.name.length), "notes.txt".length) + 2;
  const row = (k: string, v: string) => `  ${k.padEnd(14)}${v}`;

  return [
    "Trellis Studio — modeler handoff",
    "================================",
    row("Asset:", assetBaseName(rec.name)),
    row("Generated:", new Date(rec.ts).toLocaleString()),
    row("Exported:", new Date().toLocaleString()),
    "",
    "Generation (same image + seed + settings reproduces the same model)",
    row(
      "Model:",
      p.model === "pixal3d"
        ? `Pixal3D (TencentARC, TRELLIS.2 fine-tune), run locally with trellis.cpp; image FOV ${+fovOf(p).toFixed(2)} deg`
        : "TRELLIS.2 (Microsoft), run locally with trellis.cpp",
    ),
    row("Resolution:", `${p.resolution}${p.resolution > 512 ? " (cascade)" : ""}`),
    row("Seed:", String(p.seed)),
    row("Preset:", presetName(p)),
    row("Faces:", `${info.faces.toLocaleString()} (target: ${formatFaces(budgetOf(p).faces)})`),
    row("Atlas:", atlas),
    row("Textures:", webpTextures ? "WebP" : "PNG"),
    row("Background:", `${bgLabel(p.bgRemoval)} (${p.bgRemoval})`),
    row("UV unwrap:", p.uv),
    "",
    "Files",
    ...files.map((f) => `  ${f.name.padEnd(nameWidth)}${describeFile(f.name)}`),
    `  ${"notes.txt".padEnd(nameWidth)}this file`,
    "",
    "Notes for the modeler",
    "  - This is AI-generated REFERENCE, not a production mesh: dense, triangulated,",
    "    auto-generated UVs. Retopologize and re-UV it for Unreal.",
    "  - The model only saw the source image. Sides it couldn't see (back, underside)",
    "    are its best guess; where they disagree, the source image wins.",
    "  - Scale: the mesh is normalized to fit a 1 x 1 x 1 cube (glTF units = meters).",
    "    Real-world size: ______________________",
    "  - Axes: glTF convention, +Y up. The 0° view in the sheets looks at the model",
    "    from its +Z side; check it against the source image to confirm the front.",
    "    Unreal is Z-up and in centimeters; its glTF importer converts both.",
    "  - Base color can contain some shading from the source image's lighting.",
    ...(webpTextures
      ? [
          "  - Textures are WebP and the GLB REQUIRES the EXT_texture_webp extension: tools",
          "    without it will refuse to open the GLB. The extracted texture files open in",
          "    any image editor. Ask for a PNG re-export if the GLB won't open.",
        ]
      : []),
    "",
  ].join("\n");
}
