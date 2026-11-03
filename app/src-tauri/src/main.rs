// Trellis Studio — Tauri v2 desktop shell around the trellis-server image→3D
// pipeline. Reads the installer-written config.json, launches & supervises the
// server, and exposes a few commands to the web UI.
#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

mod config;
mod server;

use server::ServerState;
use tauri::Manager;

#[tauri::command]
fn get_config() -> Option<config::Config> {
    config::load().map(|mut c| {
        if c.output_dir.trim().is_empty() {
            c.output_dir = config::default_output_dir();
        }
        c
    })
}

#[tauri::command]
fn save_config(config: config::Config) -> Result<(), String> {
    config::save(&config)
}

#[tauri::command]
fn default_output_dir() -> String {
    config::default_output_dir()
}

/// Resolve the output dir (creating it), return the full path for `name` so the UI
/// can write the GLB there via the fs plugin.
#[tauri::command]
fn output_path(name: String) -> Result<String, String> {
    let dir = config::resolve_output_dir()?;
    Ok(dir.join(name).to_string_lossy().into_owned())
}

/// Open `path` in the OS file browser (Explorer / Finder / xdg-open).
fn open_in_file_browser(path: &std::path::Path) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    let mut cmd = std::process::Command::new("explorer");
    #[cfg(target_os = "macos")]
    let mut cmd = std::process::Command::new("open");
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut cmd = std::process::Command::new("xdg-open");
    cmd.arg(path);
    // explorer.exe returns a non-zero exit code even on success; spawn() ignores it.
    cmd.spawn().map(|_| ()).map_err(|e| e.to_string())
}

/// Open the output directory in the OS file browser.
/// AppData\Local is awkward to reach in Explorer, so this button matters.
#[tauri::command]
fn open_output_dir() -> Result<(), String> {
    let dir = config::resolve_output_dir()?;
    open_in_file_browser(&dir)
}

/// Turn a UI-supplied name into a single safe folder name: letters, digits,
/// `-`, `_`, `.` and spaces only, no leading/trailing dots or spaces (Windows
/// rejects those), at most 80 chars. Never empty.
fn safe_folder_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | ' ') { c } else { '_' })
        .take(80)
        .collect();
    let trimmed = cleaned.trim_matches(|c| c == '.' || c == ' ');
    if trimmed.is_empty() { "asset".to_string() } else { trimmed.to_string() }
}

/// Create (if needed) and return `<output dir>/handoff/<name>`: the folder a
/// modeler handoff package is written into. The UI then writes each file there
/// with the fs plugin, so no extra fs permissions are needed.
#[tauri::command]
fn handoff_dir(name: String) -> Result<String, String> {
    let dir = config::resolve_output_dir()?
        .join("handoff")
        .join(safe_folder_name(&name));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir.to_string_lossy().into_owned())
}

/// Open a folder in the OS file browser, but only one inside the output dir
/// (e.g. a handoff folder), so the UI can't be used to open arbitrary paths.
#[tauri::command]
fn reveal_in_output(path: String) -> Result<(), String> {
    let root = config::resolve_output_dir()?
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let target = std::path::PathBuf::from(&path)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    if !target.starts_with(&root) {
        return Err("path is outside the output folder".to_string());
    }
    // Open the path as given: on Windows canonicalize() adds a \\?\ prefix
    // that Explorer doesn't accept.
    open_in_file_browser(std::path::Path::new(&path))
}

/// The logs directory path (whether or not it exists yet).
#[tauri::command]
fn logs_dir() -> String {
    config::logs_dir()
}

/// Open the logs directory in the OS file browser, creating it if needed.
#[tauri::command]
fn open_logs_dir() -> Result<(), String> {
    let dir = config::resolve_logs_dir()?;
    open_in_file_browser(&dir)
}

/// Path of the current/last server launch's log file, if any.
#[tauri::command]
fn current_log_path(state: tauri::State<ServerState>) -> Option<String> {
    server::log_path(state.inner())
}

#[tauri::command]
fn restart_server(app: tauri::AppHandle, state: tauri::State<ServerState>) -> Result<(), String> {
    let cfg = config::load().ok_or("no config.json found")?;
    // Explicit restart: the user just changed settings, so never reuse a stale
    // server on the port — spawn fresh so the new config actually takes effect.
    server::start(&app, &cfg, state.inner(), false)
}

#[tauri::command]
fn server_running(state: tauri::State<ServerState>) -> bool {
    server::is_running(state.inner())
}

fn main() {
    // WebKitGTK ≥2.42 + the NVIDIA proprietary driver (and some other GPU/driver
    // combos) render a blank white window through the DMA-BUF path, and can even
    // crash the compositor on launch. Disabling that path fixes it with no
    // downside for this app's simple WebGL preview. Only set it if the user
    // hasn't already, so an explicit override still wins. (Linux only.)
    #[cfg(target_os = "linux")]
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(ServerState::default())
        .invoke_handler(tauri::generate_handler![
            get_config,
            save_config,
            default_output_dir,
            output_path,
            open_output_dir,
            logs_dir,
            open_logs_dir,
            current_log_path,
            restart_server,
            server_running,
            handoff_dir,
            reveal_in_output
        ])
        .setup(|app| {
            // Auto-launch the server if the installer already wrote a usable config.
            if let Some(cfg) = config::load() {
                if !cfg.server_bin.is_empty() {
                    let state = app.state::<ServerState>();
                    // Autostart may adopt a server already on the port (manual
                    // launch / pre-fix orphan) rather than fail to bind.
                    if let Err(e) = server::start(app.handle(), &cfg, state.inner(), true) {
                        eprintln!("[studio] server autostart failed: {e}");
                    }
                }
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                server::stop(window.state::<ServerState>().inner());
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building Trellis Studio")
        .run(|app, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                server::stop(app.state::<ServerState>().inner());
            }
        });
}

#[cfg(test)]
mod tests {
    use super::safe_folder_name;

    #[test]
    fn keeps_ordinary_names() {
        assert_eq!(safe_folder_name("knight_1024_mjx2k"), "knight_1024_mjx2k");
        assert_eq!(safe_folder_name("Iron Chest v2"), "Iron Chest v2");
    }

    #[test]
    fn neutralises_path_tricks() {
        assert_eq!(safe_folder_name("../../etc"), "_.._etc");
        assert_eq!(safe_folder_name(r"C:\Windows"), "C__Windows");
        assert_eq!(safe_folder_name("a/b"), "a_b");
    }

    #[test]
    fn never_empty_and_no_trailing_dot_or_space() {
        assert_eq!(safe_folder_name(""), "asset");
        assert_eq!(safe_folder_name("..."), "asset");
        assert_eq!(safe_folder_name(" name. "), "name");
    }

    #[test]
    fn caps_length() {
        assert_eq!(safe_folder_name(&"x".repeat(200)).len(), 80);
    }
}
