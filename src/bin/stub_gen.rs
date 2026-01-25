#[cfg(feature = "python-binding")]
use pyo3_stub_gen::Result;

#[cfg(feature = "python-binding")]
fn main() -> Result<()> {
    traj_dist_rs::binding::distance::sspd::stub_info()?.generate()?;
    Ok(())
}

#[cfg(not(feature = "python-binding"))]
fn main() {
    eprintln!("Error: stub_gen requires the 'python-binding' feature to be enabled.");
    eprintln!("Run with: cargo run --features python-binding --bin stub_gen");
    std::process::exit(1);
}
