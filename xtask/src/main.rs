#![forbid(unsafe_code)]
//! Repository automation wrapper for sim-femm generated documentation.

mod host_blind;
mod index_check;
mod simdoc;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let result = match args.get(1).map(String::as_str) {
        Some("index-check") => index_check::run(args),
        Some("check-host-blind") => host_blind::run(),
        _ => simdoc::run(args),
    };
    if let Err(err) = result {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
