//! Enforces the host-blind boundary for published FEMM crates.

use std::{fs, path::Path};

const FORBIDDEN: &[&str] = &[
    "std::time::Instant",
    "std::time::SystemTime",
    "std::env::",
    "std::fs::",
    "std::process::",
];

pub fn run() -> Result<(), String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("missing repository root")?;
    let mut violations = Vec::new();
    visit(&root.join("crates"), &mut |path, text| {
        if path.extension().is_some_and(|extension| extension == "rs") {
            for pattern in FORBIDDEN {
                if text.contains(pattern) {
                    violations.push(format!("{}: forbidden host API {pattern}", path.display()));
                }
            }
        }
    })?;
    if violations.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "FEMM must remain host-blind:\n{}",
            violations.join("\n")
        ))
    }
}

fn visit(directory: &Path, inspect: &mut impl FnMut(&Path, &str)) -> Result<(), String> {
    for entry in
        fs::read_dir(directory).map_err(|error| format!("read {}: {error}", directory.display()))?
    {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_dir() {
            visit(&path, inspect)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let text = fs::read_to_string(&path)
                .map_err(|error| format!("read {}: {error}", path.display()))?;
            inspect(&path, &text);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn published_femm_crates_are_host_blind() {
        super::run().unwrap();
    }

    #[test]
    fn scanner_patterns_cover_clocks_files_environment_and_processes() {
        assert_eq!(
            super::FORBIDDEN,
            &[
                "std::time::Instant",
                "std::time::SystemTime",
                "std::env::",
                "std::fs::",
                "std::process::"
            ]
        );
    }
}
