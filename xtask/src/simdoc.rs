// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Thin simdoc launcher: defers to the isolated simdoc encoder.
//! index-check and check-host-blind stay in main.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn run(args: Vec<String>) -> Result<(), String> {
    let program = args.first().map(String::as_str).unwrap_or("xtask");
    if args.get(1).map(String::as_str) != Some("simdoc") {
        return Err(format!("usage: {program} simdoc [--check]"));
    }

    let root = env::current_dir().map_err(|err| format!("current dir: {err}"))?;
    let manifest = locate_simdoc_manifest(&root)?;
    let mut command = Command::new("cargo");
    command.args(["run", "--locked", "--offline", "--manifest-path"]);
    command.arg(manifest);
    command.args(["--quiet", "--", "simdoc", "--repo-root"]);
    command.arg(&root);
    for arg in args.iter().skip(2) {
        command.arg(arg);
    }

    let status = command
        .status()
        .map_err(|err| format!("run isolated simdoc encoder: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "isolated simdoc encoder failed with status {status}"
        ))
    }
}

fn locate_simdoc_manifest(repo_root: &Path) -> Result<PathBuf, String> {
    if let Ok(path) = env::var("SIMDOC_TOOLING_MANIFEST") {
        // CI and release generation set this to the tooling workspace
        // manifest. That manifest refuses the simdoc verb.
        return isolated_manifest_from_hint(Path::new(&path));
    }
    locate_sibling_simdoc(repo_root)
}

fn locate_sibling_simdoc(repo_root: &Path) -> Result<PathBuf, String> {
    let isolated = repo_root
        .parent()
        .unwrap_or(repo_root)
        .join("sim-tooling")
        .join("crates")
        .join("simdoc")
        .join("Cargo.toml");
    if isolated.is_file() {
        return Ok(isolated);
    }
    Err(
        "set SIMDOC_TOOLING_MANIFEST to the sim-tooling checkout or crates/simdoc/Cargo.toml"
            .to_owned(),
    )
}

fn isolated_manifest_from_hint(hint: &Path) -> Result<PathBuf, String> {
    let isolated_rel = Path::new("crates").join("simdoc").join("Cargo.toml");
    let candidates = [
        hint.to_path_buf(),
        hint.join("Cargo.toml"),
        hint.join(&isolated_rel),
        hint.parent().unwrap_or(hint).join(&isolated_rel),
    ];
    for candidate in candidates {
        if candidate.is_file() && candidate.ends_with(&isolated_rel) {
            return Ok(candidate);
        }
    }
    Err(format!(
        "SIMDOC_TOOLING_MANIFEST={} does not locate crates/simdoc/Cargo.toml",
        hint.display()
    ))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn isolated_rel() -> PathBuf {
        Path::new("crates").join("simdoc").join("Cargo.toml")
    }

    fn scratch_root() -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let root = env::temp_dir().join(format!(
            "sim-femm-simdoc-{}-{}-{}",
            std::process::id(),
            nanos,
            seq
        ));
        fs::create_dir_all(&root).expect("scratch");
        root
    }

    fn write_isolated(tooling: &Path) -> PathBuf {
        let manifest = tooling.join(isolated_rel());
        fs::create_dir_all(manifest.parent().expect("parent")).expect("dirs");
        fs::write(&manifest, "[package]\nname = \"simdoc\"\n").expect("manifest");
        fs::write(tooling.join("Cargo.toml"), "[workspace]\n").expect("workspace");
        manifest
    }

    #[test]
    fn tooling_hints_resolve_to_the_isolated_encoder() {
        let tooling = scratch_root();
        let selected = write_isolated(&tooling);
        assert_eq!(
            isolated_manifest_from_hint(&tooling.join("Cargo.toml")).expect("workspace hint"),
            selected
        );
        assert_eq!(
            isolated_manifest_from_hint(&selected).expect("isolated hint"),
            selected
        );
        assert_eq!(
            isolated_manifest_from_hint(&tooling).expect("checkout hint"),
            selected
        );
        fs::remove_dir_all(tooling).expect("cleanup");
    }

    #[test]
    fn missing_tooling_hint_is_an_error() {
        let root = scratch_root();
        let missing = root.join("no-such-tooling").join("Cargo.toml");
        let error = isolated_manifest_from_hint(&missing).expect_err("missing hint");
        assert!(error.contains("does not locate crates/simdoc/Cargo.toml"));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn sibling_checkout_resolves_when_the_encoder_exists() {
        let parent = scratch_root();
        let manifest = write_isolated(&parent.join("sim-tooling"));
        let repo = parent.join("sim-femm");
        fs::create_dir_all(&repo).expect("repo");
        assert_eq!(locate_sibling_simdoc(&repo).expect("sibling"), manifest);
        fs::remove_dir_all(parent).expect("cleanup");
    }

    #[test]
    fn missing_sibling_checkout_is_an_error() {
        let parent = scratch_root();
        let repo = parent.join("sim-femm");
        fs::create_dir_all(&repo).expect("repo");
        let error = locate_sibling_simdoc(&repo).expect_err("missing sibling");
        assert!(error.contains("set SIMDOC_TOOLING_MANIFEST"));
        fs::remove_dir_all(parent).expect("cleanup");
    }
}
