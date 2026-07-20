// Supervises the resident trellis-server child process: spawn it from the
// configured binary path, forward its stdout/stderr lines to the UI as
// `server-log` events (so the UI can show live stage progress), and make sure
// it dies with the app.

use std::io::{BufRead, BufReader, Read};
use std::net::{TcpStream, ToSocketAddrs};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;
use tauri::{AppHandle, Emitter};

use crate::config::Config;

#[derive(Default)]
pub struct ServerState(pub Mutex<Option<Child>>);

/// Is something already accepting connections on host:port? Used to reuse a
/// server left running by a previously-crashed app instance instead of spawning
/// a duplicate that would fail to bind the port.
fn port_open(host: &str, port: u16) -> bool {
    match format!("{host}:{port}").to_socket_addrs() {
        Ok(mut addrs) => addrs
            .next()
            .map(|a| TcpStream::connect_timeout(&a, Duration::from_millis(300)).is_ok())
            .unwrap_or(false),
        Err(_) => false,
    }
}

fn pipe<R: Read + Send + 'static>(app: AppHandle, reader: R) {
    std::thread::spawn(move || {
        let buf = BufReader::new(reader);
        for line in buf.lines().map_while(Result::ok) {
            let _ = app.emit("server-log", line);
        }
    });
}

/// (Re)start the server from the given config. Stops any existing child first.
pub fn start(app: &AppHandle, cfg: &Config, state: &ServerState) -> Result<(), String> {
    stop(state);
    if cfg.server_bin.is_empty() {
        return Err("server binary is not configured".to_string());
    }

    // Reuse a server already bound to this port (e.g. one orphaned by a previous
    // crash) rather than spawn a duplicate that can't bind. We don't own it, so we
    // leave the tracked child as None and won't try to kill it on exit.
    if port_open(&cfg.host, cfg.port) {
        let _ = app.emit(
            "server-log",
            format!("[studio] reusing server already on {}:{}", cfg.host, cfg.port),
        );
        return Ok(());
    }

    let mut cmd = Command::new(&cfg.server_bin);
    cmd.arg("--models")
        .arg(&cfg.models_dir)
        .arg("--gpu")
        .arg(cfg.gpu.to_string())
        .arg("--host")
        .arg(&cfg.host)
        .arg("--port")
        .arg(cfg.port.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Don't flash a console window on Windows.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to launch {}: {e}", cfg.server_bin))?;

    if let Some(out) = child.stdout.take() {
        pipe(app.clone(), out);
    }
    if let Some(err) = child.stderr.take() {
        pipe(app.clone(), err);
    }

    *state.0.lock().unwrap() = Some(child);
    let _ = app.emit(
        "server-log",
        format!("[studio] launched {} on {}:{}", cfg.server_bin, cfg.host, cfg.port),
    );
    Ok(())
}

pub fn stop(state: &ServerState) {
    if let Some(mut child) = state.0.lock().unwrap().take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

pub fn is_running(state: &ServerState) -> bool {
    let mut guard = state.0.lock().unwrap();
    match guard.as_mut() {
        Some(child) => matches!(child.try_wait(), Ok(None)),
        None => false,
    }
}
