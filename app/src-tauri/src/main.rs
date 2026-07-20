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
    config::load()
}

#[tauri::command]
fn save_config(config: config::Config) -> Result<(), String> {
    config::save(&config)
}

#[tauri::command]
fn restart_server(app: tauri::AppHandle, state: tauri::State<ServerState>) -> Result<(), String> {
    let cfg = config::load().ok_or("no config.json found")?;
    server::start(&app, &cfg, state.inner())
}

#[tauri::command]
fn server_running(state: tauri::State<ServerState>) -> bool {
    server::is_running(state.inner())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(ServerState::default())
        .invoke_handler(tauri::generate_handler![
            get_config,
            save_config,
            restart_server,
            server_running
        ])
        .setup(|app| {
            // Auto-launch the server if the installer already wrote a usable config.
            if let Some(cfg) = config::load() {
                if !cfg.server_bin.is_empty() {
                    let state = app.state::<ServerState>();
                    if let Err(e) = server::start(app.handle(), &cfg, state.inner()) {
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
