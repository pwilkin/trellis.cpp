import { defineConfig } from "vite";

// Tauri expects a fixed dev port and a relative asset base so the built bundle
// loads correctly from the app's embedded file server. `clearScreen: false`
// keeps Rust compiler output visible during `tauri dev`.
export default defineConfig({
  base: "./",
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
  build: {
    target: "es2021",
    minify: "esbuild",
    sourcemap: false,
  },
});
