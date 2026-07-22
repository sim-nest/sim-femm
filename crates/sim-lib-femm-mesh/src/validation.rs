#![forbid(unsafe_code)]
//! Pre-mesh validation of a FEMM model.
//!
//! Checks that a model declares its materials, parameters, boundaries, and
//! sources consistently before it is lowered and meshed.

use std::collections::BTreeSet;

use sim_kernel::{Cx, Expr, Symbol};
use sim_lib_femm_core::{FemmError, FemmResult, Formulation, ParamSet};
use sim_lib_femm_geometry::{Geometry2, LoweredGeometry2};
use sim_lib_femm_material::{Boundary, BoundaryKind, Material, MeshPolicy, Source};

use crate::implementation::FemmModel;

/// Validates a [`FemmModel`] before it is lowered and meshed.
///
/// Checks that materials are present, every region references a declared
/// material, sources and boundaries name existing entities, and all parameter
/// expressions reference declared parameters or intrinsic coordinates. Returns
/// the first inconsistency as a [`FemmError`].
///
/// [`FemmModel`]: crate::FemmModel
pub fn validate_model(model: &FemmModel, params: &ParamSet) -> FemmResult<()> {
    if model.materials.is_empty() {
        return Err(FemmError::MissingMaterial(
            "model has no materials".to_owned(),
        ));
    }
    let declared = declared_symbols(model, params);
    for input in &model.inputs {
        let declared = params.get(&input.name).is_some() || input.default.is_some();
        if !declared {
            return Err(FemmError::UnknownFemmParameter(input.name.to_string()));
        }
    }
    let material_names = model
        .materials
        .iter()
        .map(|material| material.name.clone())
        .collect::<BTreeSet<_>>();
    for label in &model.geometry.labels {
        if !material_names.contains(&label.material) {
            return Err(FemmError::MissingMaterial(label.material.to_string()));
        }
    }
    for region in &model.geometry.analytic {
        let material = analytic_region_name(region);
        if !material_names.contains(material) {
            return Err(FemmError::MissingMaterial(material.to_string()));
        }
    }
    model.geometry.validate_supported()?;
    validate_supported_boundaries(model)?;
    validate_source_regions(model)?;
    validate_boundary_refs(model)?;
    validate_expr_refs(model, &declared)?;
    Ok(())
}

pub(crate) fn validate_lowered_geometry(
    model: &FemmModel,
    lowered: &LoweredGeometry2,
) -> FemmResult<()> {
    if matches!(model.formulation, Formulation::Axisymmetric)
        && lowered.nodes.iter().any(|point| point[0] < 0.0)
    {
        return Err(FemmError::InvalidGeometry(
            "axisymmetric geometry crosses r = 0".to_owned(),
        ));
    }
    Ok(())
}

fn validate_supported_boundaries(model: &FemmModel) -> FemmResult<()> {
    for boundary in &model.boundaries {
        if boundary.kind != BoundaryKind::Dirichlet {
            return Err(FemmError::InvalidGeometry(format!(
                "unsupported boundary kind {}",
                boundary.kind
            )));
        }
    }
    Ok(())
}

fn validate_source_regions(model: &FemmModel) -> FemmResult<()> {
    let region_names = region_names(model);
    for source in &model.sources {
        let region = source_region(source);
        if !region_names.contains(region) {
            return Err(FemmError::InvalidGeometry(format!(
                "unknown source region {region}"
            )));
        }
    }
    Ok(())
}

fn validate_boundary_refs(model: &FemmModel) -> FemmResult<()> {
    let boundary_names = model
        .boundaries
        .iter()
        .map(|boundary| boundary.name.clone())
        .collect::<BTreeSet<_>>();
    for segment in &model.geometry.segments {
        if let Some(boundary) = &segment.boundary
            && !boundary_names.contains(boundary)
        {
            return Err(FemmError::InvalidGeometry(format!(
                "unknown segment boundary {boundary}"
            )));
        }
    }
    for region in &model.geometry.analytic {
        if let sim_lib_femm_geometry::AnalyticRegion2::OuterBox { boundary, .. } = region
            && !boundary_names.contains(boundary)
        {
            return Err(FemmError::InvalidGeometry(format!(
                "unknown outer-box boundary {boundary}"
            )));
        }
    }
    Ok(())
}

fn declared_symbols(model: &FemmModel, params: &ParamSet) -> BTreeSet<Symbol> {
    model
        .inputs
        .iter()
        .map(|input| input.name.clone())
        .chain(params.symbols())
        .collect()
}

fn region_names(model: &FemmModel) -> BTreeSet<Symbol> {
    model
        .geometry
        .labels
        .iter()
        .map(|label| label.name.clone())
        .chain(
            model
                .geometry
                .analytic
                .iter()
                .map(analytic_region_name)
                .cloned(),
        )
        .collect()
}

fn analytic_region_name(region: &sim_lib_femm_geometry::AnalyticRegion2) -> &Symbol {
    match region {
        sim_lib_femm_geometry::AnalyticRegion2::Rect { name, .. }
        | sim_lib_femm_geometry::AnalyticRegion2::Circle { name, .. }
        | sim_lib_femm_geometry::AnalyticRegion2::Polygon { name, .. }
        | sim_lib_femm_geometry::AnalyticRegion2::OuterBox { name, .. } => name,
    }
}

fn source_region(source: &Source) -> &Symbol {
    match source {
        Source::CurrentDensity { region, .. }
        | Source::CircuitCoil { region, .. }
        | Source::ChargeDensity { region, .. }
        | Source::HeatSource { region, .. } => region,
    }
}

fn validate_expr_refs(model: &FemmModel, declared: &BTreeSet<Symbol>) -> FemmResult<()> {
    for input in &model.inputs {
        if let Some(default) = &input.default {
            validate_value_ref(default, &input.name)?;
        }
    }
    for expr in geometry_exprs(&model.geometry)
        .into_iter()
        .chain(material_exprs(&model.materials))
        .chain(boundary_exprs(&model.boundaries))
        .chain(source_exprs(&model.sources))
        .chain(mesh_policy_exprs(&model.mesh_policy))
        .chain(model.frequency_hz.iter())
        .chain(model.depth.iter())
        .chain(model.solve_policy.iter())
        .chain(model.outputs.iter().map(|output| &output.query))
    {
        validate_expr(expr, declared)?;
    }
    Ok(())
}

fn validate_value_ref(value: &sim_kernel::Value, input: &Symbol) -> FemmResult<()> {
    let mut cx = Cx::new(
        std::sync::Arc::new(sim_kernel::EagerPolicy),
        std::sync::Arc::new(sim_kernel::DefaultFactory),
    );
    value
        .object()
        .display(&mut cx)
        .map(|_| ())
        .map_err(|err| FemmError::InvalidGeometry(format!("bad default for {input}: {err}")))
}

fn validate_expr(expr: &Expr, declared: &BTreeSet<Symbol>) -> FemmResult<()> {
    match expr {
        Expr::Symbol(symbol) | Expr::Local(symbol) => {
            if declared.contains(symbol) || is_intrinsic_coord(symbol) {
                Ok(())
            } else {
                Err(FemmError::UnknownFemmParameter(symbol.to_string()))
            }
        }
        Expr::List(items) | Expr::Vector(items) | Expr::Set(items) | Expr::Block(items) => {
            validate_exprs(items.iter(), declared)
        }
        Expr::Map(entries) => {
            for (key, value) in entries {
                validate_expr(key, declared)?;
                validate_expr(value, declared)?;
            }
            Ok(())
        }
        Expr::Call { args, .. } => validate_exprs(args.iter(), declared),
        Expr::Infix { left, right, .. } => {
            validate_expr(left, declared)?;
            validate_expr(right, declared)
        }
        Expr::Prefix { arg, .. }
        | Expr::Postfix { arg, .. }
        | Expr::Extension { payload: arg, .. } => validate_expr(arg, declared),
        Expr::Quote { .. } => Ok(()),
        Expr::Annotated { expr, annotations } => {
            validate_expr(expr, declared)?;
            for (_, annotation) in annotations {
                validate_expr(annotation, declared)?;
            }
            Ok(())
        }
        Expr::Nil | Expr::Bool(_) | Expr::Number(_) | Expr::String(_) | Expr::Bytes(_) => Ok(()),
    }
}

fn validate_exprs<'a>(
    exprs: impl Iterator<Item = &'a Expr>,
    declared: &BTreeSet<Symbol>,
) -> FemmResult<()> {
    for expr in exprs {
        validate_expr(expr, declared)?;
    }
    Ok(())
}

fn is_intrinsic_coord(symbol: &Symbol) -> bool {
    matches!(symbol.name.as_ref(), "x" | "y" | "r" | "z" | "t")
}

fn geometry_exprs(geometry: &Geometry2) -> Vec<&Expr> {
    let mut out = Vec::new();
    for node in &geometry.nodes {
        out.extend(node.xy.iter());
    }
    for arc in &geometry.arcs {
        out.push(&arc.angle_deg);
    }
    for label in &geometry.labels {
        out.extend(label.at.iter());
    }
    for region in &geometry.analytic {
        match region {
            sim_lib_femm_geometry::AnalyticRegion2::Rect { xy, wh, .. } => {
                out.extend(xy.iter());
                out.extend(wh.iter());
            }
            sim_lib_femm_geometry::AnalyticRegion2::Circle { center, radius, .. } => {
                out.extend(center.iter());
                out.push(radius);
            }
            sim_lib_femm_geometry::AnalyticRegion2::Polygon { points, .. } => {
                for point in points {
                    out.extend(point.iter());
                }
            }
            sim_lib_femm_geometry::AnalyticRegion2::OuterBox { margin, .. } => {
                out.push(margin);
            }
        }
    }
    out
}

fn material_exprs(materials: &[Material]) -> Vec<&Expr> {
    let mut out = Vec::new();
    for material in materials {
        out.extend(material.mu_r.iter());
        out.extend(material.nu_of_b2.iter());
        out.extend(material.epsilon_r.iter());
        out.extend(material.sigma.iter());
        out.extend(material.thermal_k.iter());
        out.extend(material.heat_source.iter());
        if let Some(remanence) = &material.remanence {
            out.extend(remanence.iter());
        }
    }
    out
}

fn boundary_exprs(boundaries: &[Boundary]) -> Vec<&Expr> {
    boundaries.iter().map(|boundary| &boundary.value).collect()
}

fn source_exprs(sources: &[Source]) -> Vec<&Expr> {
    let mut out = Vec::new();
    for source in sources {
        match source {
            Source::CurrentDensity { value, .. }
            | Source::ChargeDensity { value, .. }
            | Source::HeatSource { value, .. } => out.push(value),
            Source::CircuitCoil { turns, current, .. } => {
                out.push(turns);
                out.push(current);
            }
        }
    }
    out
}

fn mesh_policy_exprs(policy: &MeshPolicy) -> Vec<&Expr> {
    policy
        .max_area
        .iter()
        .chain(policy.min_angle_deg.iter())
        .collect()
}
