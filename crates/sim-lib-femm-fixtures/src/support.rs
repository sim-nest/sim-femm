use sim_kernel::{Expr, Symbol};
use sim_lib_femm_material::Material;

pub(crate) fn num(text: &str) -> Expr {
    sim_value::build::num_q(Some("numbers"), "f64", text)
}

pub(crate) fn air() -> Material {
    Material {
        name: Symbol::new("air"),
        mu_r: Some(num("1.0")),
        nu_of_b2: None,
        epsilon_r: Some(num("1.0")),
        sigma: None,
        thermal_k: Some(num("1.0")),
        heat_source: None,
        remanence: None,
    }
}
