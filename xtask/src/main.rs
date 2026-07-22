#![forbid(unsafe_code)]
//! Repository automation wrapper for sim-femm generated documentation.

mod index_check;
mod simdoc;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let result = match args.get(1).map(String::as_str) {
        Some("index-check") => index_check::run(args),
        _ => simdoc::run(args),
    };
    if let Err(err) = result {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
