// config.json is the contract between the installer (writer) and the app
// (reader). In an installed build it lives at <config_dir>/trellis-studio/ —
// ~/.config/trellis-studio/ on Linux, %APPDATA%\trellis-studio\ on Windows.
//
// PORTABLE MODE: if a `portable.dat` marker sits next to the executable (as in
// the portable .zip / .tar.gz), everything lives next to the exe instead —
// config + generated files under <exe-dir>/data/, and the server + models are
// picked up from <exe-dir>/runtime/ and <exe-dir>/models/ when present. Nothing
// touches the system config/data dirs, so the unzipped folder is self-contained.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Config {
    #[serde(rename = "serverBin", default)]
    pub server_bin: String,
    #[serde(rename = "modelsDir", default)]
    pub models_dir: String,
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default)]
    pub gpu: i32,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    /// Where generated GLBs are auto-saved. Empty => use `default_output_dir()`.
    #[serde(rename = "outputDir", default)]
    pub output_dir: String,
}

fn default_backend() -> String {
    "unknown".to_string()
}
fn default_host() -> String {
    "127.0.0.1".to_string()
}
fn default_port() -> u16 {
    8080
}

/// The portable root (the executable's directory) when a `portable.dat` marker
/// is present next to it; otherwise None (installed mode). AppImages run from a
/// temp mount with no marker, so they correctly stay in installed mode.
fn portable_root() -> Option<PathBuf> {
    let dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    dir.join("portable.dat").exists().then_some(dir)
}

fn server_bin_name() -> &'static str {
    if cfg!(windows) {
        "trellis-server.exe"
    } else {
        "trellis-server"
    }
}

/// Default location for generated GLBs: <exe>/data/output in portable mode, else
/// <local-data>/trellis-studio/output (%LOCALAPPDATA%\... / ~/.local/share/...).
pub fn default_output_dir() -> String {
    if let Some(root) = portable_root() {
        return root.join("data").join("output").to_string_lossy().into_owned();
    }
    dirs::data_local_dir()
        .map(|d| d.join("trellis-studio").join("output"))
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The effective output dir (config value or default), created if missing.
pub fn resolve_output_dir() -> Result<PathBuf, String> {
    let dir = load()
        .map(|c| c.output_dir)
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(default_output_dir);
    if dir.is_empty() {
        return Err("could not determine an output directory".to_string());
    }
    let p = PathBuf::from(dir);
    std::fs::create_dir_all(&p).map_err(|e| e.to_string())?;
    Ok(p)
}

pub fn config_path() -> Option<PathBuf> {
    if let Some(root) = portable_root() {
        return Some(root.join("data").join("config.json"));
    }
    dirs::config_dir().map(|d| d.join("trellis-studio").join("config.json"))
}

fn load_from_file() -> Option<Config> {
    let p = config_path()?;
    let s = std::fs::read_to_string(p).ok()?;
    // Tolerate a UTF-8 BOM: some Windows editors / PowerShell's `Set-Content
    // -Encoding UTF8` prepend one, and serde_json won't parse past it.
    let s = s.strip_prefix('\u{feff}').unwrap_or(&s);
    serde_json::from_str(s).ok()
}

pub fn load() -> Option<Config> {
    if let Some(cfg) = load_from_file() {
        return Some(cfg);
    }
    // Portable fallback: no config.json yet, but if the portable folder already
    // has a runtime/ + models/ layout, synthesize a usable config from it so the
    // app works the moment you drop those in and launch.
    let root = portable_root()?;
    let server = root.join("runtime").join(server_bin_name());
    server.exists().then(|| Config {
        server_bin: server.to_string_lossy().into_owned(),
        models_dir: root.join("models").to_string_lossy().into_owned(),
        backend: default_backend(),
        gpu: 0,
        host: default_host(),
        port: default_port(),
        output_dir: String::new(),
    })
}

pub fn save(cfg: &Config) -> Result<(), String> {
    let p = config_path().ok_or("no config directory available")?;
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let s = serde_json::to_string_pretty(cfg).map_err(|e| e.to_string())?;
    std::fs::write(p, s).map_err(|e| e.to_string())
}
