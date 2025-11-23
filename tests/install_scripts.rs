use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn fixture(path: &str) -> PathBuf {
    fs::canonicalize(PathBuf::from(path)).expect("fixture path")
}

#[test]
fn install_sh_succeeds_with_valid_checksum() {
    let tar = fixture("tests/fixtures/install/coding-agent-search-vtest-linux-x86_64.tar.gz");
    let checksum = fs::read_to_string(
        "tests/fixtures/install/coding-agent-search-vtest-linux-x86_64.tar.gz.sha256",
    )
    .unwrap()
    .trim()
    .to_string();
    let dest = tempfile::TempDir::new().unwrap();

    let status = Command::new("bash")
        .arg("install.sh")
        .arg("--version")
        .arg("vtest")
        .arg("--dest")
        .arg(dest.path())
        .arg("--easy-mode")
        .env("ARTIFACT_URL", format!("file://{}", tar.display()))
        .env("CHECKSUM", checksum)
        .status()
        .expect("run install.sh");

    assert!(status.success());
    let bin = dest.path().join("coding-agent-search");
    assert!(bin.exists());
    let output = Command::new(&bin).output().expect("run installed bin");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("fixture-linux"));
}

#[test]
fn install_sh_fails_with_bad_checksum() {
    let tar = fixture("tests/fixtures/install/coding-agent-search-vtest-linux-x86_64.tar.gz");
    let dest = tempfile::TempDir::new().unwrap();

    let status = Command::new("bash")
        .arg("install.sh")
        .arg("--version")
        .arg("vtest")
        .arg("--dest")
        .arg(dest.path())
        .arg("--easy-mode")
        .env("ARTIFACT_URL", format!("file://{}", tar.display()))
        .env("CHECKSUM", "deadbeef")
        .status()
        .expect("run install.sh");

    assert!(!status.success());
    assert!(!dest.path().join("coding-agent-search").exists());
}

fn find_powershell() -> Option<String> {
    for candidate in [&"pwsh", &"powershell"] {
        if let Ok(path) = which::which(candidate) {
            return Some(path.to_string_lossy().into_owned());
        }
    }
    None
}

#[test]
fn install_ps1_succeeds_with_valid_checksum() {
    if !cfg!(target_os = "windows") {
        eprintln!("skipping powershell test: non-windows runner");
        return;
    }
    let Some(ps) = find_powershell() else {
        eprintln!("skipping powershell test: pwsh not found");
        return;
    };

    let zip = fixture("tests/fixtures/install/coding-agent-search-vtest-windows-x86_64.zip");
    let checksum = fs::read_to_string(
        "tests/fixtures/install/coding-agent-search-vtest-windows-x86_64.zip.sha256",
    )
    .unwrap()
    .trim()
    .to_string();
    let dest = tempfile::TempDir::new().unwrap();

    let status = Command::new(ps)
        .arg("-NoProfile")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-File")
        .arg("install.ps1")
        .arg("-Version")
        .arg("vtest")
        .arg("-Dest")
        .arg(dest.path())
        .arg("-Checksum")
        .arg(&checksum)
        .arg("-EasyMode")
        .arg("-ArtifactUrl")
        .arg(format!("file://{}", zip.display()))
        .status()
        .expect("run install.ps1");

    assert!(status.success());
    let bin = dest.path().join("coding-agent-search.exe");
    assert!(bin.exists());
    let content = fs::read_to_string(&bin).unwrap();
    assert!(content.contains("fixture-windows"));
}

#[test]
fn install_ps1_fails_with_bad_checksum() {
    if !cfg!(target_os = "windows") {
        eprintln!("skipping powershell test: non-windows runner");
        return;
    }
    let Some(ps) = find_powershell() else {
        eprintln!("skipping powershell test: pwsh not found");
        return;
    };

    let zip = fixture("tests/fixtures/install/coding-agent-search-vtest-windows-x86_64.zip");
    let dest = tempfile::TempDir::new().unwrap();

    let status = Command::new(ps)
        .arg("-NoProfile")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-File")
        .arg("install.ps1")
        .arg("-Version")
        .arg("vtest")
        .arg("-Dest")
        .arg(dest.path())
        .arg("-Checksum")
        .arg("deadbeef")
        .arg("-EasyMode")
        .arg("-ArtifactUrl")
        .arg(format!("file://{}", zip.display()))
        .status()
        .expect("run install.ps1");

    assert!(!status.success());
    assert!(!dest.path().join("coding-agent-search.exe").exists());
}
