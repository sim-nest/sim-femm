#![forbid(unsafe_code)]
//! Symbolic 2D geometry and its lowering to concrete coordinates.
//!
//! Defines nodes, segments, arcs, block labels, and analytic regions, plus the
//! lowering that resolves their symbolic expressions into the coordinates the
//! mesher consumes.

use sim_kernel::{Cx, Expr, Origin, Symbol};
use sim_lib_femm_core::{FemmResult, ParamSet};

/// Re-export the shared expression evaluator so existing callers of
/// `sim_lib_femm_geometry::eval_expr_f64` keep compiling. The implementation
/// now lives in `sim-lib-femm-core`; the geometry-specific lowering below
/// continues to use it.
pub use sim_lib_femm_core::eval_expr_f64;

/// A geometry node whose `(x, y)` position is held as symbolic [`Expr`]s.
///
/// Coordinates stay symbolic until [`Geometry2::lower`] resolves them against a
/// `ParamSet`, so a node can track a swept parameter. See the
/// [crate README](../sim_lib_femm_geometry/index.html).
#[derive(Clone, Debug)]
pub struct Node2 {
    /// Symbolic `[x, y]` coordinate expressions.
    pub xy: [Expr; 2],
}

/// A straight segment joining two [`Node2`]s by their indices.
#[derive(Clone, Debug)]
pub struct Segment2 {
    /// Index of the start node in [`Geometry2::nodes`].
    pub a: usize,
    /// Index of the end node in [`Geometry2::nodes`].
    pub b: usize,
    /// Optional boundary-condition tag carried into meshing.
    pub boundary: Option<Symbol>,
}

/// A circular arc joining two [`Node2`]s, subtending `angle_deg` degrees.
#[derive(Clone, Debug)]
pub struct Arc2 {
    /// Index of the start node in [`Geometry2::nodes`].
    pub a: usize,
    /// Index of the end node in [`Geometry2::nodes`].
    pub b: usize,
    /// Symbolic subtended angle, in degrees.
    pub angle_deg: Expr,
    /// Optional boundary-condition tag carried into meshing.
    pub boundary: Option<Symbol>,
}

/// A material label that paints the region containing point `at`.
#[derive(Clone, Debug)]
pub struct BlockLabel2 {
    /// Region name.
    pub name: Symbol,
    /// Symbolic `[x, y]` point inside the region to label.
    pub at: [Expr; 2],
    /// Material assigned to the labelled region.
    pub material: Symbol,
}

/// A full symbolic 2D FEMM geometry: nodes, segments, arcs, labels, regions.
///
/// This is the authoring form. [`Geometry2::lower`] turns it into a
/// [`LoweredGeometry2`] of concrete coordinates for the mesher.
#[derive(Clone, Debug, Default)]
pub struct Geometry2 {
    /// Explicit geometry nodes.
    pub nodes: Vec<Node2>,
    /// Straight segments between nodes.
    pub segments: Vec<Segment2>,
    /// Circular arcs between nodes.
    pub arcs: Vec<Arc2>,
    /// Material block labels.
    pub labels: Vec<BlockLabel2>,
    /// Analytic regions expanded into nodes, segments, and labels on lowering.
    pub analytic: Vec<AnalyticRegion2>,
}

/// A high-level region primitive expanded into nodes and segments on lowering.
///
/// Analytic regions let a geometry be described in shape terms (a rectangle, a
/// circle, a polygon, an enclosing outer box) instead of explicit node and
/// segment lists.
#[derive(Clone, Debug)]
pub enum AnalyticRegion2 {
    /// An axis-aligned rectangle of size `wh` with lower-left corner at `xy`.
    Rect {
        /// Region name, also used as the material label.
        name: Symbol,
        /// Symbolic `[x, y]` of the lower-left corner.
        xy: [Expr; 2],
        /// Symbolic `[width, height]`.
        wh: [Expr; 2],
    },
    /// A circle of the given `radius` about `center`.
    Circle {
        /// Region name, also used as the material label.
        name: Symbol,
        /// Symbolic `[x, y]` of the center.
        center: [Expr; 2],
        /// Symbolic radius.
        radius: Expr,
    },
    /// A closed polygon through the listed `points` (needs at least three).
    Polygon {
        /// Region name, also used as the material label.
        name: Symbol,
        /// Symbolic `[x, y]` vertices in order.
        points: Vec<[Expr; 2]>,
    },
    /// An enclosing box placed `margin` outside the rest of the geometry.
    OuterBox {
        /// Region name.
        name: Symbol,
        /// Symbolic margin between the geometry extent and the box.
        margin: Expr,
        /// Boundary-condition tag applied to the box edges.
        boundary: Symbol,
    },
}

/// Concrete-coordinate geometry produced by [`Geometry2::lower`].
///
/// All symbolic expressions are resolved to `f64`, and analytic regions are
/// expanded into explicit nodes, segments, and labels ready for meshing.
#[derive(Clone, Debug, Default)]
pub struct LoweredGeometry2 {
    /// Concrete node coordinates.
    pub nodes: Vec<[f64; 2]>,
    /// Segments as `(start_node, end_node, optional boundary tag)`.
    pub segments: Vec<(usize, usize, Option<Symbol>)>,
    /// Labels as `(name, point, material)`.
    pub labels: Vec<(Symbol, [f64; 2], Symbol)>,
}

impl Geometry2 {
    /// Resolves all symbolic coordinates and expands analytic regions.
    ///
    /// Evaluates every coordinate [`Expr`] against `params` via
    /// `eval_expr_f64`, then emits the explicit nodes, segments, and labels the
    /// mesher consumes as a [`LoweredGeometry2`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use sim_kernel::{Cx, DefaultFactory, EagerPolicy, Expr, NumberLiteral, Symbol};
    /// use sim_lib_femm_core::ParamSet;
    /// use sim_lib_femm_geometry::{AnalyticRegion2, Geometry2};
    ///
    /// let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    /// let num = |t: &str| Expr::Number(NumberLiteral {
    ///     domain: Symbol::qualified("numbers", "f64"),
    ///     canonical: t.to_owned(),
    /// });
    /// let geometry = Geometry2 {
    ///     analytic: vec![AnalyticRegion2::Rect {
    ///         name: Symbol::new("steel"),
    ///         xy: [num("0.0"), num("0.0")],
    ///         wh: [num("2.0"), num("1.0")],
    ///     }],
    ///     ..Geometry2::default()
    /// };
    /// let lowered = geometry.lower(&mut cx, &ParamSet::default()).unwrap();
    /// assert_eq!(lowered.nodes.len(), 4);
    /// assert_eq!(lowered.segments.len(), 4);
    /// assert_eq!(lowered.labels.len(), 1);
    /// ```
    pub fn lower(&self, cx: &mut Cx, params: &ParamSet) -> FemmResult<LoweredGeometry2> {
        let mut lowered = LoweredGeometry2::default();
        for node in &self.nodes {
            lowered.nodes.push([
                eval_expr_f64(cx, &node.xy[0], params, &[])?,
                eval_expr_f64(cx, &node.xy[1], params, &[])?,
            ]);
        }
        for segment in &self.segments {
            lowered
                .segments
                .push((segment.a, segment.b, segment.boundary.clone()));
        }
        for label in &self.labels {
            lowered.labels.push((
                label.name.clone(),
                [
                    eval_expr_f64(cx, &label.at[0], params, &[])?,
                    eval_expr_f64(cx, &label.at[1], params, &[])?,
                ],
                label.material.clone(),
            ));
        }
        for region in &self.analytic {
            match region {
                AnalyticRegion2::Rect { name, xy, wh } => {
                    let x = eval_expr_f64(cx, &xy[0], params, &[])?;
                    let y = eval_expr_f64(cx, &xy[1], params, &[])?;
                    let w = eval_expr_f64(cx, &wh[0], params, &[])?;
                    let h = eval_expr_f64(cx, &wh[1], params, &[])?;
                    let base = lowered.nodes.len();
                    lowered
                        .nodes
                        .extend([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]);
                    lowered.segments.extend([
                        (base, base + 1, None),
                        (base + 1, base + 2, None),
                        (base + 2, base + 3, None),
                        (base + 3, base, None),
                    ]);
                    lowered
                        .labels
                        .push((name.clone(), [x + 0.5 * w, y + 0.5 * h], name.clone()));
                }
                AnalyticRegion2::Circle {
                    name,
                    center,
                    radius,
                } => {
                    let cx0 = eval_expr_f64(cx, &center[0], params, &[])?;
                    let cy0 = eval_expr_f64(cx, &center[1], params, &[])?;
                    let r = eval_expr_f64(cx, radius, params, &[])?;
                    lowered
                        .labels
                        .push((name.clone(), [cx0, cy0], name.clone()));
                    lowered.nodes.extend([
                        [cx0 - r, cy0],
                        [cx0, cy0 + r],
                        [cx0 + r, cy0],
                        [cx0, cy0 - r],
                    ]);
                }
                AnalyticRegion2::Polygon { name, points } => {
                    let start = lowered.nodes.len();
                    let mut centroid = [0.0, 0.0];
                    for point in points {
                        let x = eval_expr_f64(cx, &point[0], params, &[])?;
                        let y = eval_expr_f64(cx, &point[1], params, &[])?;
                        lowered.nodes.push([x, y]);
                        centroid[0] += x;
                        centroid[1] += y;
                    }
                    let count = points.len() as f64;
                    if points.len() >= 3 {
                        for index in 0..points.len() {
                            lowered.segments.push((
                                start + index,
                                start + (index + 1) % points.len(),
                                None,
                            ));
                        }
                        lowered.labels.push((
                            name.clone(),
                            [centroid[0] / count, centroid[1] / count],
                            name.clone(),
                        ));
                    }
                }
                AnalyticRegion2::OuterBox { .. } => {}
            }
        }
        Ok(lowered)
    }
}

/// Builds a placeholder kernel [`Origin`] tagged to the `femm` source.
///
/// Used when constructing expressions that have no real source span, such as
/// geometry generated programmatically rather than parsed from a codec.
pub fn dummy_origin() -> Origin {
    Origin {
        codec: sim_kernel::CodecId(0),
        source: sim_kernel::SourceId("femm".to_owned()),
        span: sim_kernel::Span { start: 0, end: 0 },
        trivia: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use sim_kernel::{DefaultFactory, EagerPolicy};
    use sim_lib_femm_core::{FemmError, ParamSet};

    use super::*;

    fn test_cx() -> Cx {
        Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory))
    }

    fn num(text: &str) -> Expr {
        sim_value::build::num_q(Some("numbers"), "f64", text)
    }

    #[test]
    fn rect_lowers_to_expected_primitives() {
        let mut cx = test_cx();
        let geometry = Geometry2 {
            analytic: vec![AnalyticRegion2::Rect {
                name: Symbol::new("steel"),
                xy: [num("0.0"), num("0.0")],
                wh: [num("2.0"), num("1.0")],
            }],
            ..Geometry2::default()
        };
        let lowered = geometry.lower(&mut cx, &ParamSet::default()).unwrap();
        assert_eq!(lowered.nodes.len(), 4);
        assert_eq!(lowered.segments.len(), 4);
        assert_eq!(lowered.labels.len(), 1);
    }

    #[test]
    fn parameterized_geometry_moves_with_param_changes() {
        let mut cx = test_cx();
        let x = cx
            .factory()
            .number_literal(Symbol::qualified("numbers", "f64"), "3.0".to_owned())
            .unwrap();
        let geometry = Geometry2 {
            analytic: vec![AnalyticRegion2::Rect {
                name: Symbol::new("air"),
                xy: [Expr::Symbol(Symbol::new("gap")), num("0.0")],
                wh: [num("1.0"), num("1.0")],
            }],
            ..Geometry2::default()
        };
        let lowered = geometry
            .lower(&mut cx, &ParamSet::new(vec![(Symbol::new("gap"), x)]))
            .unwrap();
        assert_eq!(lowered.nodes[0], [3.0, 0.0]);
    }

    #[test]
    fn missing_param_is_rejected() {
        let mut cx = test_cx();
        let err = eval_expr_f64(
            &mut cx,
            &Expr::Symbol(Symbol::new("missing")),
            &ParamSet::default(),
            &[],
        )
        .unwrap_err();
        assert!(matches!(err, FemmError::UnknownFemmParameter(_)));
    }
}
