use tempfile::TempDir;

/// E2E: watch-mode smoke. Touch a fixture file and ensure incremental re-index logs fire.
#[test]
fn watch_mode_reindexes_on_file_change() {
    // Temp sandbox to isolate all filesystem access
    let sandbox = TempDir::new().expect("temp dir");
    let data_dir = sandbox.path().join("data");
    let home_dir = sandbox.path().join("home");
    let xdg_data = sandbox.path().join("xdg-data");
    let xdg_config = sandbox.path().join("xdg-config");
    std::fs::create_dir_all(&data_dir).expect("data dir");
    std::fs::create_dir_all(&home_dir).expect("home dir");
    std::fs::create_dir_all(&xdg_data).expect("xdg data");
    std::fs::create_dir_all(&xdg_config).expect("xdg config");

    // Seed a tiny connector fixture under Codex path so watch can detect
    let codex_root = data_dir.join(".codex/sessions");
    std::fs::create_dir_all(&codex_root).expect("codex root");
    let rollout = codex_root.join("rollout-1.jsonl");
    std::fs::write(
        &rollout,
        r#"{"role":"user","content":"hello","createdAt":1700000000000}"#,
    )
    .expect("write rollout");

    // Start watch in background: cass index --watch --data-dir <tmp>
    // Resolve cass binary path from env if available, else fallback to cargo_bin!
    let cass_bin = std::env::var("CARGO_BIN_EXE_cass")
        .ok()
        .unwrap_or_else(|| env!("CARGO_BIN_EXE_cass").to_string());
    let output = std::process::Command::new(cass_bin)
        .arg("index")
        .arg("--watch")
        .arg("--watch-once")
        .arg(rollout.to_string_lossy().to_string())
        .arg("--data-dir")
        .arg(&data_dir)
        // Point all XDG/HOME roots at the sandbox to avoid scanning host data
        .env("HOME", &home_dir)
        .env("XDG_DATA_HOME", &xdg_data)
        .env("XDG_CONFIG_HOME", &xdg_config)
        .env("CODEX_HOME", data_dir.join(".codex"))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .expect("run watch");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "watch run failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    // Verify watch_state.json was updated for Codex connector
    let watch_state_path = data_dir.join("watch_state.json");
    let contents = std::fs::read_to_string(&watch_state_path).unwrap_or_else(|_| {
        panic!(
            "missing watch_state at {}. stdout: {stdout}, stderr: {stderr}",
            watch_state_path.display()
        )
    });
    let map: std::collections::HashMap<String, i64> =
        serde_json::from_str(&contents).expect("parse watch_state");
    let ts = map.get("Codex").copied().unwrap_or(0);
    assert!(ts > 0, "expected Codex entry in watch_state, got {:?}", map);
}
