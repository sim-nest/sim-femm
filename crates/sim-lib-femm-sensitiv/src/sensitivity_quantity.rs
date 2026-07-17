use sim_kernel::{Cx, Symbol};
use sim_lib_femm_core::{FemmError, FemmResult};
use sim_lib_femm_post::{Excitation, FemmSolution, QuantitySpec};
use sim_lib_numbers_ad::{Dual, Scalarish};

use crate::sensitivity_solve::{axisymmetric_weight, dual_geom};
use crate::sensitivity_types::{DiffMesh, DiffSolution, DualGeom};

pub(crate) fn quantity_derivative(
    cx: &mut Cx,
    diff: &DiffSolution,
    spec: &QuantitySpec,
    excitation: &Excitation,
) -> FemmResult<f64> {
    validate_diff_solution(diff)?;
    let value = match spec {
        QuantitySpec::Energy { region } | QuantitySpec::Coenergy { region } => {
            region_energy(diff, region.as_ref())?.d[0]
        }
        QuantitySpec::ForceY { region } => region_force_y(diff, region)?.d[0],
        QuantitySpec::Torque { region, center } => {
            let force = region_force_y(diff, region)?;
            let centroid = region_centroid_x(diff, region)?;
            (force * (centroid - Dual::cst(center[0]))).d[0]
        }
        QuantitySpec::FluxLinkage { .. } => {
            let current = require_current(excitation, "flux linkage")?;
            // lambda = 2W/I  =>  d(lambda)/dp = (2/I) dW/dp for a fixed current.
            2.0 * region_energy(diff, None)?.d[0] / current
        }
        QuantitySpec::Inductance { .. } => {
            let current = require_current(excitation, "inductance")?;
            // L = 2W/I^2  =>  dL/dp = (2/I^2) dW/dp for a fixed current.
            2.0 * region_energy(diff, None)?.d[0] / (current * current)
        }
        QuantitySpec::Capacitance { .. } => {
            let potential = require_potential(excitation, "capacitance")?;
            // C = 2W/V^2  =>  dC/dp = (2/V^2) dW/dp for a fixed potential.
            2.0 * region_energy(diff, None)?.d[0] / (potential * potential)
        }
        QuantitySpec::JouleLoss { region } => joule_loss(diff, region.as_ref())?.d[0],
        QuantitySpec::FieldAt { field, points } => {
            let point = sim_lib_femm_core::decode_point2(points)?;
            sample_named_field(diff, field, point)?.d[0]
        }
        QuantitySpec::Custom { .. } => {
            return Err(FemmError::SensitivityUnavailable(
                "custom quantity uses expression adjoint path".to_owned(),
            ));
        }
    };
    let _ = cx;
    Ok(value)
}

/// Requires a nonzero coil current, mirroring the forward `quantity` guard.
///
/// The exact derivative of a current-referenced quantity (`lambda = 2W/I`,
/// `L = 2W/I^2`) is undefined without a drive, so it fails closed exactly like
/// the forward value rather than reporting a spurious sensitivity.
fn require_current(excitation: &Excitation, quantity: &str) -> FemmResult<f64> {
    let current = excitation
        .current()
        .ok_or_else(|| FemmError::FieldOutOfDomain(format!("{quantity} needs a coil current")))?;
    if current == 0.0 {
        return Err(FemmError::InvalidGeometry(format!(
            "{quantity} undefined at zero current"
        )));
    }
    Ok(current)
}

/// Requires a nonzero applied potential, mirroring the forward `quantity` guard.
fn require_potential(excitation: &Excitation, quantity: &str) -> FemmResult<f64> {
    let potential = excitation.potential().ok_or_else(|| {
        FemmError::FieldOutOfDomain(format!("{quantity} needs an applied potential"))
    })?;
    if potential == 0.0 {
        return Err(FemmError::InvalidGeometry(format!(
            "{quantity} undefined at zero potential"
        )));
    }
    Ok(potential)
}

fn region_energy(diff: &DiffSolution, region: Option<&Symbol>) -> FemmResult<Dual<1>> {
    let mut total = Dual::cst(0.0);
    let mut matched = false;
    for (index, tri) in diff.solution.mesh.tri.iter().enumerate() {
        if !region_matches(&diff.solution, index, region) {
            continue;
        }
        matched = true;
        let geom = dual_geom_for_solution(diff, *tri)?;
        let measure = geom.area * axisymmetric_weight(&geom, &diff.solution.formulation)?;
        let mean = node_mean(diff, *tri)?;
        total = total + measure * mean * mean * Dual::cst(0.5);
    }
    ensure_region_match(region, matched)?;
    Ok(total)
}

fn region_force_y(diff: &DiffSolution, region: &Symbol) -> FemmResult<Dual<1>> {
    let mut total = Dual::cst(0.0);
    let mut matched = false;
    for (index, tri) in diff.solution.mesh.tri.iter().enumerate() {
        if !region_matches(&diff.solution, index, Some(region)) {
            continue;
        }
        matched = true;
        let geom = dual_geom_for_solution(diff, *tri)?;
        let measure = geom.area * axisymmetric_weight(&geom, &diff.solution.formulation)?;
        let grad = sample_gradient_dual(diff, *tri)?;
        total = total - measure * grad[1] * grad[1] * Dual::cst(0.5);
    }
    ensure_region_match(Some(region), matched)?;
    Ok(total)
}

fn region_centroid_x(diff: &DiffSolution, region: &Symbol) -> FemmResult<Dual<1>> {
    let mut weighted = Dual::cst(0.0);
    let mut measure_total = Dual::cst(0.0);
    let mut matched = false;
    for (index, tri) in diff.solution.mesh.tri.iter().enumerate() {
        if !region_matches(&diff.solution, index, Some(region)) {
            continue;
        }
        matched = true;
        let geom = dual_geom_for_solution(diff, *tri)?;
        let measure = geom.area * axisymmetric_weight(&geom, &diff.solution.formulation)?;
        let centroid = (geom.xy[0][0] + geom.xy[1][0] + geom.xy[2][0]) / Dual::cst(3.0);
        weighted = weighted + centroid * measure;
        measure_total = measure_total + measure;
    }
    ensure_region_match(Some(region), matched)?;
    Ok(weighted / measure_total)
}

fn joule_loss(diff: &DiffSolution, region: Option<&Symbol>) -> FemmResult<Dual<1>> {
    let mut total = Dual::cst(0.0);
    let mut matched = false;
    for (index, tri) in diff.solution.mesh.tri.iter().enumerate() {
        if !region_matches(&diff.solution, index, region) {
            continue;
        }
        matched = true;
        let geom = dual_geom_for_solution(diff, *tri)?;
        let measure = geom.area * axisymmetric_weight(&geom, &diff.solution.formulation)?;
        let grad = sample_gradient_dual(diff, *tri)?;
        total = total + measure * (grad[0] * grad[0] + grad[1] * grad[1]);
    }
    ensure_region_match(region, matched)?;
    Ok(total)
}

fn sample_named_field(diff: &DiffSolution, field: &Symbol, point: [f64; 2]) -> FemmResult<Dual<1>> {
    match field.name.as_ref() {
        "potential" | "a" | "v" => sample_potential_dual(diff, point),
        "bx" | "ex" => Ok(sample_gradient_dual(diff, locate_triangle(diff, point)?.0)?[0]),
        "by" | "ey" => Ok(sample_gradient_dual(diff, locate_triangle(diff, point)?.0)?[1]),
        "bmag" | "emag" | "heat-flux-mag" => {
            let grad = sample_gradient_dual(diff, locate_triangle(diff, point)?.0)?;
            Ok((grad[0] * grad[0] + grad[1] * grad[1]).sqrt())
        }
        _ => Err(FemmError::FieldOutOfDomain(format!(
            "unknown field {field}"
        ))),
    }
}

fn sample_potential_dual(diff: &DiffSolution, point: [f64; 2]) -> FemmResult<Dual<1>> {
    let (tri, bary) = locate_triangle(diff, point)?;
    let values = node_values(diff, tri)?;
    Ok((0..3)
        .map(|index| bary[index] * values[index])
        .fold(Dual::cst(0.0), |acc, value| acc + value))
}

fn sample_gradient_dual(diff: &DiffSolution, tri: [u32; 3]) -> FemmResult<[Dual<1>; 2]> {
    let geom = dual_geom_for_solution(diff, tri)?;
    let values = node_values(diff, tri)?;
    Ok([
        geom.grad
            .iter()
            .zip(values)
            .map(|(grad, value)| grad[0] * value)
            .fold(Dual::cst(0.0), |acc, value| acc + value),
        geom.grad
            .iter()
            .zip(values)
            .map(|(grad, value)| grad[1] * value)
            .fold(Dual::cst(0.0), |acc, value| acc + value),
    ])
}

fn locate_triangle(diff: &DiffSolution, point: [f64; 2]) -> FemmResult<([u32; 3], [Dual<1>; 3])> {
    validate_diff_solution(diff)?;
    for tri in &diff.solution.mesh.tri {
        let geom = dual_geom_for_solution(diff, *tri)?;
        let bary = barycentric_dual(&geom, point);
        if bary.iter().all(|value| value.v >= -1.0e-9) {
            return Ok((*tri, bary));
        }
    }
    Err(FemmError::FieldOutOfDomain(format!(
        "point ({}, {})",
        point[0], point[1]
    )))
}

fn barycentric_dual(geom: &DualGeom, point: [f64; 2]) -> [Dual<1>; 3] {
    let px = Dual::cst(point[0]);
    let py = Dual::cst(point[1]);
    let area = geom.area * Dual::cst(2.0);
    let l0 = ((geom.xy[1][0] - px) * (geom.xy[2][1] - py)
        - (geom.xy[2][0] - px) * (geom.xy[1][1] - py))
        / area;
    let l1 = ((geom.xy[2][0] - px) * (geom.xy[0][1] - py)
        - (geom.xy[0][0] - px) * (geom.xy[2][1] - py))
        / area;
    [l0, l1, Dual::cst(1.0) - l0 - l1]
}

fn dual_geom_for_solution(diff: &DiffSolution, tri: [u32; 3]) -> FemmResult<DualGeom> {
    dual_geom(
        &DiffMesh {
            mesh: diff.solution.mesh.clone(),
            dxy: diff.dxy.clone(),
        },
        tri,
    )
}

fn validate_diff_solution(diff: &DiffSolution) -> FemmResult<()> {
    diff.solution.validate()?;
    if diff.du.len() != diff.solution.u.len() {
        return Err(FemmError::InvalidGeometry(format!(
            "sensitivity solution has {} derivative values but {} solution values",
            diff.du.len(),
            diff.solution.u.len()
        )));
    }
    if diff.dxy.len() != diff.solution.mesh.xy.len() {
        return Err(FemmError::InvalidGeometry(format!(
            "sensitivity mesh has {} derivative nodes but {} mesh nodes",
            diff.dxy.len(),
            diff.solution.mesh.xy.len()
        )));
    }
    Ok(())
}

fn node_values(diff: &DiffSolution, tri: [u32; 3]) -> FemmResult<[Dual<1>; 3]> {
    let mut values = [Dual::cst(0.0); 3];
    for local in 0..3 {
        let index = tri[local] as usize;
        let v = *diff.solution.u.get(index).ok_or_else(|| {
            FemmError::InvalidGeometry(format!(
                "triangle node {} has no solution value",
                tri[local]
            ))
        })?;
        let d = *diff.du.get(index).ok_or_else(|| {
            FemmError::InvalidGeometry(format!(
                "triangle node {} has no solution derivative",
                tri[local]
            ))
        })?;
        values[local] = Dual { v, d: [d] };
    }
    Ok(values)
}

fn node_mean(diff: &DiffSolution, tri: [u32; 3]) -> FemmResult<Dual<1>> {
    Ok(node_values(diff, tri)?
        .into_iter()
        .fold(Dual::cst(0.0), |acc, value| acc + value)
        / Dual::cst(3.0))
}

fn region_matches(solution: &FemmSolution, index: usize, region: Option<&Symbol>) -> bool {
    region.is_none_or(|region| {
        solution
            .mesh
            .elem_region
            .get(index)
            .is_some_and(|name| name == region)
    })
}

fn ensure_region_match(region: Option<&Symbol>, matched: bool) -> FemmResult<()> {
    if let Some(region) = region
        && !matched
    {
        return Err(FemmError::FieldOutOfDomain(format!(
            "missing region {region}"
        )));
    }
    Ok(())
}
