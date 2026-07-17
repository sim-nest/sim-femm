use sim_kernel::{Cx, Symbol};
use sim_lib_femm_core::{FemmError, FemmResult, ParamSet};
use sim_lib_femm_geometry::{AnalyticRegion2, Geometry2};
use sim_lib_femm_mesh::{FemMesh2, FemmModel, MeshedModel, Mesher};
use sim_lib_numbers_ad::Dual;

use crate::expr_eval::eval_expr_dual;
use crate::sensitivity_types::DiffMesh;

pub(crate) fn differentiated_mesh(
    cx: &mut Cx,
    model: &FemmModel,
    params: &ParamSet,
    wrt: &Symbol,
) -> FemmResult<DiffMesh> {
    let predicted = sim_lib_femm_mesh::DeterministicMesher::new().mesh(cx, model, params)?;
    let lowered = lower_geometry(cx, &model.geometry, params, wrt)?;
    if lowered.nodes.len() < 3 {
        return Err(FemmError::InvalidGeometry(
            "need at least three lowered nodes".to_owned(),
        ));
    }
    let mut tri = Vec::new();
    if lowered.nodes.len() == 4 {
        tri.extend([[0, 1, 2], [0, 2, 3]]);
    } else {
        for index in 1..lowered.nodes.len() - 1 {
            tri.push([0_u32, index as u32, (index + 1) as u32]);
        }
    }
    let region = lowered
        .labels
        .first()
        .map(|(name, _, _)| name.clone())
        .ok_or_else(|| {
            FemmError::InvalidGeometry("lowered geometry has no material labels".to_owned())
        })?;
    let mesh = FemMesh2 {
        xy: lowered.nodes,
        tri: tri.clone(),
        elem_region: vec![region; tri.len()],
        edge_boundary: lowered
            .segments
            .into_iter()
            .filter_map(|(a, b, boundary)| boundary.map(|name| (a as u32, b as u32, name)))
            .collect(),
    };
    let meshed = MeshedModel {
        model_id: model.id,
        params: params.clone(),
        mesh,
        diagnostics: Vec::new(),
    };
    meshed.validate_against(model)?;
    if predicted.mesh.tri.len() != meshed.mesh.tri.len()
        || predicted.mesh.xy.len() != meshed.mesh.xy.len()
    {
        return Err(FemmError::SensitivityUnavailable(
            "differentiated mesh topology mismatch".to_owned(),
        ));
    }
    Ok(DiffMesh {
        mesh: meshed.mesh,
        dxy: lowered.dxy,
    })
}

struct LoweredDual {
    nodes: Vec<[f64; 2]>,
    dxy: Vec<[f64; 2]>,
    segments: Vec<(usize, usize, Option<Symbol>)>,
    labels: Vec<(Symbol, [f64; 2], Symbol)>,
}

fn lower_geometry(
    cx: &mut Cx,
    geometry: &Geometry2,
    params: &ParamSet,
    wrt: &Symbol,
) -> FemmResult<LoweredDual> {
    geometry.validate_supported()?;
    let mut lowered = LoweredDual {
        nodes: Vec::new(),
        dxy: Vec::new(),
        segments: Vec::new(),
        labels: Vec::new(),
    };
    for node in &geometry.nodes {
        lowered.push_point(cx, params, wrt, &node.xy)?;
    }
    for segment in &geometry.segments {
        lowered
            .segments
            .push((segment.a, segment.b, segment.boundary.clone()));
    }
    for label in &geometry.labels {
        lowered.labels.push((
            label.name.clone(),
            [
                eval_expr_dual(cx, &label.at[0], params, Some(wrt), &[])?.v,
                eval_expr_dual(cx, &label.at[1], params, Some(wrt), &[])?.v,
            ],
            label.material.clone(),
        ));
    }
    for region in &geometry.analytic {
        lowered.lower_region(cx, params, wrt, region)?;
    }
    Ok(lowered)
}

impl LoweredDual {
    fn lower_region(
        &mut self,
        cx: &mut Cx,
        params: &ParamSet,
        wrt: &Symbol,
        region: &AnalyticRegion2,
    ) -> FemmResult<()> {
        match region {
            AnalyticRegion2::Rect { name, xy, wh } => {
                let x = eval_expr_dual(cx, &xy[0], params, Some(wrt), &[])?;
                let y = eval_expr_dual(cx, &xy[1], params, Some(wrt), &[])?;
                let w = eval_expr_dual(cx, &wh[0], params, Some(wrt), &[])?;
                let h = eval_expr_dual(cx, &wh[1], params, Some(wrt), &[])?;
                let base = self.nodes.len();
                self.push_dual_point(x, y);
                self.push_dual_point(x + w, y);
                self.push_dual_point(x + w, y + h);
                self.push_dual_point(x, y + h);
                self.segments.extend([
                    (base, base + 1, None),
                    (base + 1, base + 2, None),
                    (base + 2, base + 3, None),
                    (base + 3, base, None),
                ]);
                self.labels.push((
                    name.clone(),
                    [x.v + 0.5 * w.v, y.v + 0.5 * h.v],
                    name.clone(),
                ));
            }
            AnalyticRegion2::Circle {
                name,
                center,
                radius,
            } => {
                let cx0 = eval_expr_dual(cx, &center[0], params, Some(wrt), &[])?;
                let cy0 = eval_expr_dual(cx, &center[1], params, Some(wrt), &[])?;
                let r = eval_expr_dual(cx, radius, params, Some(wrt), &[])?;
                self.labels
                    .push((name.clone(), [cx0.v, cy0.v], name.clone()));
                self.push_dual_point(cx0 - r, cy0);
                self.push_dual_point(cx0, cy0 + r);
                self.push_dual_point(cx0 + r, cy0);
                self.push_dual_point(cx0, cy0 - r);
            }
            AnalyticRegion2::Polygon { name, points } => {
                let start = self.nodes.len();
                let mut centroid = [0.0, 0.0];
                for point in points {
                    self.push_point(cx, params, wrt, point)?;
                    centroid[0] += self.nodes.last().unwrap()[0];
                    centroid[1] += self.nodes.last().unwrap()[1];
                }
                if points.len() >= 3 {
                    for index in 0..points.len() {
                        self.segments.push((
                            start + index,
                            start + (index + 1) % points.len(),
                            None,
                        ));
                    }
                    let count = points.len() as f64;
                    self.labels.push((
                        name.clone(),
                        [centroid[0] / count, centroid[1] / count],
                        name.clone(),
                    ));
                }
            }
            AnalyticRegion2::OuterBox { .. } => unreachable!("validated before lowering"),
        }
        Ok(())
    }

    fn push_point(
        &mut self,
        cx: &mut Cx,
        params: &ParamSet,
        wrt: &Symbol,
        point: &[sim_kernel::Expr; 2],
    ) -> FemmResult<()> {
        let x = eval_expr_dual(cx, &point[0], params, Some(wrt), &[])?;
        let y = eval_expr_dual(cx, &point[1], params, Some(wrt), &[])?;
        self.push_dual_point(x, y);
        Ok(())
    }

    fn push_dual_point(&mut self, x: Dual<1>, y: Dual<1>) {
        self.nodes.push([x.v, y.v]);
        self.dxy.push([x.d[0], y.d[0]]);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use sim_kernel::{DefaultFactory, EagerPolicy, Expr};

    use super::*;

    fn test_cx() -> Cx {
        Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory))
    }

    #[test]
    fn sensitivity_lowering_uses_geometry_supported_shape_contract() {
        let mut cx = test_cx();
        let geometry = Geometry2 {
            analytic: vec![AnalyticRegion2::OuterBox {
                name: Symbol::new("air"),
                margin: Expr::Symbol(Symbol::new("missing")),
                boundary: Symbol::new("open"),
            }],
            ..Geometry2::default()
        };
        let result = lower_geometry(
            &mut cx,
            &geometry,
            &ParamSet::default(),
            &Symbol::new("gap"),
        );
        assert!(matches!(result, Err(FemmError::InvalidGeometry(_))));
    }
}
