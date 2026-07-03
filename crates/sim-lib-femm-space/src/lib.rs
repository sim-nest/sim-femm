#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Function and DOF spaces over the FEMM mesh.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: the per-element geometry, basis gradients,
//! and barycentric mapping that back the finite-element function space.

use sim_lib_femm_core::{FemmError, FemmResult, Formulation};
use sim_lib_femm_mesh::FemMesh2;

/// Per-element geometry for a linear triangular finite element.
///
/// The local function-space data for one mesh triangle: its vertices, signed
/// area, and the constant gradients of the three P1 (linear) nodal basis
/// functions. These back the DOF space's element-level integration. See the
/// [crate README](https://github.com/sim-nest/sim-femm) for the FEM role.
///
/// # Examples
///
/// ```
/// use sim_kernel::Symbol;
/// use sim_lib_femm_mesh::FemMesh2;
/// use sim_lib_femm_space::ElementGeom;
///
/// let mesh = FemMesh2 {
///     xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
///     tri: vec![[0, 1, 2]],
///     elem_region: vec![Symbol::new("air")],
///     edge_boundary: Vec::new(),
/// };
/// let geom = ElementGeom::from_mesh(&mesh, [0, 1, 2]).unwrap();
/// assert!((geom.area - 0.5).abs() < 1.0e-12);
/// // Barycentric coordinates of any point sum to one.
/// let lambda = geom.barycentric([0.2, 0.3]);
/// assert!((lambda.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
/// ```
#[derive(Clone, Debug)]
pub struct ElementGeom {
    /// Vertex coordinates of the triangle, one `[x, y]` per node.
    pub xy: [[f64; 2]; 3],
    /// Positive area of the triangle.
    pub area: f64,
    /// Constant gradient `[d/dx, d/dy]` of each nodal basis function.
    pub grad: [[f64; 2]; 3],
}

impl ElementGeom {
    /// Builds the element geometry for triangle `tri` of `mesh`.
    ///
    /// Computes the area and basis gradients from the three vertices. Returns
    /// [`FemmError::InvalidGeometry`] for a degenerate (zero-area) triangle.
    pub fn from_mesh(mesh: &FemMesh2, tri: [u32; 3]) -> FemmResult<Self> {
        let xy = [
            mesh.xy[tri[0] as usize],
            mesh.xy[tri[1] as usize],
            mesh.xy[tri[2] as usize],
        ];
        let area2 = (xy[1][0] - xy[0][0]) * (xy[2][1] - xy[0][1])
            - (xy[2][0] - xy[0][0]) * (xy[1][1] - xy[0][1]);
        let area = 0.5 * area2.abs();
        if area <= f64::EPSILON {
            return Err(FemmError::InvalidGeometry("degenerate triangle".to_owned()));
        }
        let denom = 2.0 * area;
        let grad = [
            [(xy[1][1] - xy[2][1]) / denom, (xy[2][0] - xy[1][0]) / denom],
            [(xy[2][1] - xy[0][1]) / denom, (xy[0][0] - xy[2][0]) / denom],
            [(xy[0][1] - xy[1][1]) / denom, (xy[1][0] - xy[0][0]) / denom],
        ];
        Ok(Self { xy, area, grad })
    }

    /// Returns the barycentric coordinates of `point` within the triangle.
    ///
    /// The three values are the P1 basis functions evaluated at `point` and sum
    /// to one; they lie in `[0, 1]` exactly when the point is inside the element.
    pub fn barycentric(&self, point: [f64; 2]) -> [f64; 3] {
        let area = self.area * 2.0;
        let lambda0 = ((self.xy[1][0] - point[0]) * (self.xy[2][1] - point[1])
            - (self.xy[2][0] - point[0]) * (self.xy[1][1] - point[1]))
            / area;
        let lambda1 = ((self.xy[2][0] - point[0]) * (self.xy[0][1] - point[1])
            - (self.xy[0][0] - point[0]) * (self.xy[2][1] - point[1]))
            / area;
        let lambda2 = 1.0 - lambda0 - lambda1;
        [lambda0, lambda1, lambda2]
    }

    /// Returns the integration weight for this element under `formulation`.
    ///
    /// Unity for planar problems; for axisymmetric problems the `2*pi*r` factor
    /// at the element centroid. Returns [`FemmError::InvalidGeometry`] if the
    /// centroidal radius is negative.
    pub fn axisymmetric_weight(&self, formulation: &Formulation) -> FemmResult<f64> {
        match formulation {
            Formulation::Planar => Ok(1.0),
            Formulation::Axisymmetric => {
                let r = (self.xy[0][0] + self.xy[1][0] + self.xy[2][0]) / 3.0;
                if r < 0.0 {
                    return Err(FemmError::InvalidGeometry("axisymmetric r < 0".to_owned()));
                }
                Ok(2.0 * std::f64::consts::PI * r)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use sim_kernel::Symbol;
    use sim_lib_femm_mesh::FemMesh2;

    use super::*;

    fn test_geom() -> ElementGeom {
        ElementGeom::from_mesh(
            &FemMesh2 {
                xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                tri: vec![[0, 1, 2]],
                elem_region: vec![Symbol::new("air")],
                edge_boundary: Vec::new(),
            },
            [0, 1, 2],
        )
        .unwrap()
    }

    #[test]
    fn basis_functions_sum_to_one() {
        let geom = test_geom();
        let lambda = geom.barycentric([0.2, 0.3]);
        assert!((lambda.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn linear_field_gradient_is_exact() {
        let geom = test_geom();
        let field = |x: f64, y: f64| 2.0 * x + 3.0 * y + 1.0;
        let u = geom.xy.map(|pt| field(pt[0], pt[1]));
        let grad_x = geom
            .grad
            .iter()
            .zip(u)
            .map(|(grad, value)| grad[0] * value)
            .sum::<f64>();
        let grad_y = geom
            .grad
            .iter()
            .zip(u)
            .map(|(grad, value)| grad[1] * value)
            .sum::<f64>();
        assert!((grad_x - 2.0).abs() < 1.0e-12);
        assert!((grad_y - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn axisymmetric_weight_rejects_negative_radius() {
        let geom = ElementGeom {
            xy: [[-1.0, 0.0], [-0.5, 0.0], [-0.5, 1.0]],
            ..test_geom()
        };
        let err = geom
            .axisymmetric_weight(&Formulation::Axisymmetric)
            .unwrap_err();
        assert!(matches!(err, FemmError::InvalidGeometry(_)));
    }
}
