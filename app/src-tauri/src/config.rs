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

pub fn config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("trellis-studio").join("config.json"))
}

pub fn load() -> Option<Config> {
    let p = config_path()?;
    let s = std::fs::read_to_string(p).ok()?;
    serde_json::from_str(&s).ok()
}

pub fn save(cfg: &Config) -> Result<(), String> {
    let p = config_path().ok_or("no config directory available")?;
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let s = serde_json::to_string_pretty(cfg).map_err(|e| e.to_string())?;
    std::fs::write(p, s).map_err(|e| e.to_string())
}
