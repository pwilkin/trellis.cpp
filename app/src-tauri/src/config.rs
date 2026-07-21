// config.json is the contract between the installer (writer) and the app
// (reader). It lives at <config_dir>/trellis-studio/config.json — i.e.
// ~/.config/trellis-studio/ on Linux, %APPDATA%\trellis-studio\ on Windows.

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

/// Default location for generated GLBs: <local-data>/trellis-studio/output
/// (%LOCALAPPDATA%\trellis-studio\output on Windows, ~/.local/share/... on Linux).
pub fn default_output_dir() -> String {
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
    dirs::config_dir().map(|d| d.join("trellis-studio").join("config.json"))
}

pub fn load() -> Option<Config> {
    let p = config_path()?;
    let s = std::fs::read_to_string(p).ok()?;
    // Tolerate a UTF-8 BOM: some Windows editors / PowerShell's `Set-Content
    // -Encoding UTF8` prepend one, and serde_json won't parse past it.
    let s = s.strip_prefix('\u{feff}').unwrap_or(&s);
    serde_json::from_str(s).ok()
}

pub fn save(cfg: &Config) -> Result<(), String> {
    let p = config_path().ok_or("no config directory available")?;
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let s = serde_json::to_string_pretty(cfg).map_err(|e| e.to_string())?;
    std::fs::write(p, s).map_err(|e| e.to_string())
}
