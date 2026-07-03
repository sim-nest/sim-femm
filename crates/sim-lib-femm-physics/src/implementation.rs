#![forbid(unsafe_code)]
//! Physics fronts for the supported FEMM formulations.
//!
//! Implements the assembly physics-front trait for the magnetostatic,
//! harmonic, electrostatic, heat, and current formulations, supplying their
//! element residuals and source terms.

use sim_lib_femm_assembly::{CoeffEval, PhysicsFront};
use sim_lib_femm_core::{FemmError, FemmResult, PhysicsKind};
use sim_lib_femm_space::ElementGeom;
use sim_lib_numbers_ad::Scalarish;

/// Electrostatic physics front (Poisson equation in electric potential).
///
/// Contributes a permittivity-weighted Laplacian stiffness term plus a charge
/// source. Implements [`PhysicsFront`] for assembly. See the
/// [crate README](https://github.com/sim-nest/sim-femm) for the FEM role.
#[derive(Clone, Debug, Default)]
pub struct ElectrostaticFront;

/// Steady heat-conduction physics front.
///
/// Contributes a thermal-conductivity-weighted Laplacian stiffness term plus a
/// heat source. Implements [`PhysicsFront`] for assembly.
#[derive(Clone, Debug, Default)]
pub struct HeatSteadyFront;

/// Steady conductive (DC current) physics front.
///
/// Contributes a conductivity-weighted Laplacian stiffness term plus a current
/// source. Implements [`PhysicsFront`] for assembly.
#[derive(Clone, Debug, Default)]
pub struct CurrentSteadyFront;

/// Magnetostatic physics front (vector potential, static field).
///
/// Contributes a reluctivity-weighted (`1 / mu_r`) Laplacian stiffness term plus
/// a current source. Implements [`PhysicsFront`] for assembly.
#[derive(Clone, Debug, Default)]
pub struct MagnetostaticFront;

/// Time-harmonic magnetics physics front (eddy-current regime).
///
/// The true eddy-current operator carries an imaginary `j * omega * sigma` term
/// that the real-valued assembly here cannot represent. Rather than fold it into
/// a real coefficient (a silently wrong answer), this front fails closed for any
/// nonzero excitation frequency via [`PhysicsFront::validate_coeff`] and reduces
/// to the linear magnetostatic reluctivity at `frequency_hz == 0`. Implements
/// [`PhysicsFront`] for assembly.
#[derive(Clone, Debug, Default)]
pub struct MagneticsHarmonicFront;

fn stiffness_residual<S: Scalarish>(elem: &ElementGeom, coeff: f64, u_e: [S; 3]) -> [S; 3] {
    let grad_u = [
        elem.grad
            .iter()
            .zip(u_e)
            .map(|(grad, u)| S::from_f64(grad[0]) * u)
            .fold(S::from_f64(0.0), |acc, value| acc + value),
        elem.grad
            .iter()
            .zip(u_e)
            .map(|(grad, u)| S::from_f64(grad[1]) * u)
            .fold(S::from_f64(0.0), |acc, value| acc + value),
    ];
    std::array::from_fn(|index| {
        let dot = grad_u[0] * S::from_f64(elem.grad[index][0])
            + grad_u[1] * S::from_f64(elem.grad[index][1]);
        dot * S::from_f64(elem.area * coeff)
    })
}

fn mass_source(elem: &ElementGeom, density: f64) -> [f64; 3] {
    [elem.area * density / 3.0; 3]
}

impl PhysicsFront for ElectrostaticFront {
    fn kind(&self) -> PhysicsKind {
        PhysicsKind::Electrostatic
    }

    fn element_residual<S: Scalarish>(
        &self,
        elem: &ElementGeom,
        u_e: [S; 3],
        coeff: &CoeffEval,
    ) -> [S; 3] {
        stiffness_residual(elem, coeff.epsilon_r, u_e)
    }

    fn source_term(&self, elem: &ElementGeom, coeff: &CoeffEval) -> [f64; 3] {
        mass_source(elem, coeff.source_density)
    }
}

impl PhysicsFront for HeatSteadyFront {
    fn kind(&self) -> PhysicsKind {
        PhysicsKind::HeatSteady
    }

    fn element_residual<S: Scalarish>(
        &self,
        elem: &ElementGeom,
        u_e: [S; 3],
        coeff: &CoeffEval,
    ) -> [S; 3] {
        stiffness_residual(elem, coeff.thermal_k, u_e)
    }

    fn source_term(&self, elem: &ElementGeom, coeff: &CoeffEval) -> [f64; 3] {
        mass_source(elem, coeff.source_density)
    }
}

impl PhysicsFront for CurrentSteadyFront {
    fn kind(&self) -> PhysicsKind {
        PhysicsKind::CurrentSteady
    }

    fn element_residual<S: Scalarish>(
        &self,
        elem: &ElementGeom,
        u_e: [S; 3],
        coeff: &CoeffEval,
    ) -> [S; 3] {
        stiffness_residual(elem, coeff.sigma.max(1.0e-12), u_e)
    }

    fn source_term(&self, elem: &ElementGeom, coeff: &CoeffEval) -> [f64; 3] {
        mass_source(elem, coeff.source_density)
    }
}

impl PhysicsFront for MagnetostaticFront {
    fn kind(&self) -> PhysicsKind {
        PhysicsKind::Magnetostatic
    }

    fn element_residual<S: Scalarish>(
        &self,
        elem: &ElementGeom,
        u_e: [S; 3],
        coeff: &CoeffEval,
    ) -> [S; 3] {
        stiffness_residual(elem, 1.0 / coeff.mu_r.max(1.0e-12), u_e)
    }

    fn source_term(&self, elem: &ElementGeom, coeff: &CoeffEval) -> [f64; 3] {
        mass_source(elem, coeff.source_density)
    }

    fn validate_coeff(&self, coeff: &CoeffEval) -> FemmResult<()> {
        if coeff.nonlinear_bh {
            return Err(FemmError::UnsupportedPhysics(
                "nonlinear B-H material requires the nonlinear solver; \
                 not supported by the linear magnetostatic front"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

impl PhysicsFront for MagneticsHarmonicFront {
    fn kind(&self) -> PhysicsKind {
        PhysicsKind::MagneticsHarmonic
    }

    fn element_residual<S: Scalarish>(
        &self,
        elem: &ElementGeom,
        u_e: [S; 3],
        coeff: &CoeffEval,
    ) -> [S; 3] {
        // The eddy-current operator adds an imaginary `j * omega * sigma` term
        // that this real-valued assembly cannot represent; folding it into a
        // real reluctivity gives a silently wrong answer. `validate_coeff`
        // therefore rejects any nonzero frequency, so the only case that reaches
        // here is `frequency_hz == 0`, where the front reduces to the linear
        // magnetostatic reluctivity.
        stiffness_residual(elem, 1.0 / coeff.mu_r.max(1.0e-12), u_e)
    }

    fn source_term(&self, elem: &ElementGeom, coeff: &CoeffEval) -> [f64; 3] {
        mass_source(elem, coeff.source_density)
    }

    fn validate_coeff(&self, coeff: &CoeffEval) -> FemmResult<()> {
        if coeff.nonlinear_bh {
            return Err(FemmError::UnsupportedPhysics(
                "nonlinear B-H material requires the nonlinear solver; \
                 not supported by the time-harmonic magnetics front"
                    .to_owned(),
            ));
        }
        if coeff.frequency_hz.abs() > 0.0 {
            return Err(FemmError::UnsupportedPhysics(
                "complex harmonic solve not supported by the real-valued \
                 magnetics-harmonic front; eddy-current coupling needs the \
                 imaginary j*omega*sigma term"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

/// Closed-form parallel-plate capacitance `epsilon * area / gap`.
///
/// Analytic reference for validating the [`ElectrostaticFront`] solution. See
/// the [crate README](https://github.com/sim-nest/sim-femm) for the FEM role.
///
/// # Examples
///
/// ```
/// use sim_lib_femm_physics::parallel_plate_capacitance;
/// assert!((parallel_plate_capacitance(8.0, 2.0, 4.0) - 4.0).abs() < 1.0e-12);
/// ```
pub fn parallel_plate_capacitance(epsilon: f64, area: f64, gap: f64) -> f64 {
    epsilon * area / gap
}

/// Closed-form 1D slab thermal resistance `length / (k * area)`.
///
/// Analytic reference for validating the [`HeatSteadyFront`] solution.
///
/// # Examples
///
/// ```
/// use sim_lib_femm_physics::slab_heat_resistance;
/// assert!((slab_heat_resistance(2.0, 4.0, 5.0) - 0.1).abs() < 1.0e-12);
/// ```
pub fn slab_heat_resistance(length: f64, k: f64, area: f64) -> f64 {
    length / (k * area)
}

/// Closed-form conductor resistance `length / (sigma * area)`.
///
/// Analytic reference for validating the [`CurrentSteadyFront`] solution.
///
/// # Examples
///
/// ```
/// use sim_lib_femm_physics::conductor_resistance;
/// assert!((conductor_resistance(10.0, 5.0, 2.0) - 1.0).abs() < 1.0e-12);
/// ```
pub fn conductor_resistance(length: f64, sigma: f64, area: f64) -> f64 {
    length / (sigma * area)
}

/// Closed-form long-solenoid axial flux density `mu0 * n * I`.
///
/// Analytic reference for validating the [`MagnetostaticFront`] solution.
///
/// # Examples
///
/// ```
/// use sim_lib_femm_physics::long_solenoid_b;
/// assert!((long_solenoid_b(2.0, 3.0, 4.0) - 24.0).abs() < 1.0e-12);
/// ```
pub fn long_solenoid_b(mu0: f64, turns_per_length: f64, current: f64) -> f64 {
    mu0 * turns_per_length * current
}

/// Closed-form time-averaged ohmic loss density `0.5 * sigma * omega * amp^2`.
///
/// Analytic reference for validating the [`MagneticsHarmonicFront`] solution.
///
/// # Examples
///
/// ```
/// use sim_lib_femm_physics::harmonic_joule_loss;
/// assert!(harmonic_joule_loss(2.0, 10.0, 3.0) > 0.0);
/// ```
pub fn harmonic_joule_loss(sigma: f64, omega: f64, amplitude: f64) -> f64 {
    0.5 * sigma * omega.abs() * amplitude * amplitude
}

#[cfg(test)]
mod tests {
    use sim_kernel::{Expr, Symbol};
    use sim_lib_femm_core::ParamSet;
    use sim_lib_femm_material::Material;

    use super::*;

    fn num(text: &str) -> Expr {
        sim_value::build::num_q(Some("numbers"), "f64", text)
    }

    fn magnetic_material(nonlinear: bool) -> Material {
        Material {
            name: Symbol::new("steel"),
            mu_r: Some(num("4000.0")),
            nu_of_b2: if nonlinear { Some(num("0.02")) } else { None },
            epsilon_r: None,
            sigma: Some(num("2.0e6")),
            thermal_k: None,
            heat_source: None,
            remanence: None,
        }
    }

    fn coeff(nonlinear_bh: bool, frequency_hz: f64) -> CoeffEval {
        CoeffEval {
            region: Symbol::new("steel"),
            material: magnetic_material(nonlinear_bh),
            params: ParamSet::default(),
            epsilon_r: 1.0,
            sigma: 2.0e6,
            thermal_k: 1.0,
            mu_r: 4000.0,
            source_density: 0.0,
            frequency_hz,
            nonlinear_bh,
        }
    }

    #[test]
    fn magnetostatic_rejects_nonlinear_bh_material() {
        // A material carrying a nonlinear B-H curve under the linear front must
        // fail closed, not silently solve linearly.
        let err = MagnetostaticFront
            .validate_coeff(&coeff(true, 0.0))
            .unwrap_err();
        assert!(matches!(err, FemmError::UnsupportedPhysics(_)));
    }

    #[test]
    fn magnetostatic_accepts_linear_material() {
        assert!(
            MagnetostaticFront
                .validate_coeff(&coeff(false, 0.0))
                .is_ok()
        );
    }

    #[test]
    fn harmonic_rejects_nonzero_frequency() {
        // The eddy-current operator is complex; a nonzero frequency cannot be
        // represented by the real-valued assembly and must fail closed.
        let err = MagneticsHarmonicFront
            .validate_coeff(&coeff(false, 60.0))
            .unwrap_err();
        assert!(matches!(err, FemmError::UnsupportedPhysics(_)));
    }

    #[test]
    fn harmonic_rejects_nonlinear_bh_material() {
        let err = MagneticsHarmonicFront
            .validate_coeff(&coeff(true, 0.0))
            .unwrap_err();
        assert!(matches!(err, FemmError::UnsupportedPhysics(_)));
    }

    #[test]
    fn harmonic_accepts_static_linear_material() {
        assert!(
            MagneticsHarmonicFront
                .validate_coeff(&coeff(false, 0.0))
                .is_ok()
        );
    }

    #[test]
    fn electrostatic_matches_parallel_plate_formula() {
        assert!((parallel_plate_capacitance(8.0, 2.0, 4.0) - 4.0).abs() < 1.0e-12);
    }

    #[test]
    fn slab_heat_matches_analytic_resistance() {
        assert!((slab_heat_resistance(2.0, 4.0, 5.0) - 0.1).abs() < 1.0e-12);
    }

    #[test]
    fn conductor_resistance_matches_analytic_result() {
        assert!((conductor_resistance(10.0, 5.0, 2.0) - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn long_solenoid_matches_mu0_n_i() {
        assert!((long_solenoid_b(2.0, 3.0, 4.0) - 24.0).abs() < 1.0e-12);
    }

    #[test]
    fn harmonic_loss_is_nonzero_at_nonzero_frequency() {
        assert!(harmonic_joule_loss(2.0, 10.0, 3.0) > 0.0);
    }
}
