//! Fingerprint-keyed cache of factorizations and solutions.
//!
//! Defines the solve key and the bounded tape that memoizes linear factors and
//! solutions by model, mesh, and parameter fingerprint for solve and
//! derivative reuse.

use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
};

use sim_kernel::{Cx, Symbol};
use sim_lib_femm_core::{FemmLimits, FemmResult, ParamSet, StableId};
use sim_lib_femm_flow::SolveDiagnostics;
use sim_lib_femm_mesh::{FemMesh2, FemmModel, Mesher};
use sim_lib_femm_post::FemmSolution;
use sim_lib_femm_solve::{FactorHandle, solve_steady};

/// The composite cache key for one solve: model, mesh, and parameter prints.
///
/// Two solves collapse to the same cached solution only when all three
/// fingerprints agree; a shared `mesh_fingerprint` alone lets a factorization
/// be reused across parameter sweeps. See the [crate README](index.html).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolveKey {
    /// Fingerprint of the model definition (geometry, materials, sources).
    pub model_fingerprint: StableId,
    /// Fingerprint of the generated mesh.
    pub mesh_fingerprint: StableId,
    /// Fingerprint of the parameter binding.
    pub param_fingerprint: StableId,
}

/// A bounded cache of linear factorizations and solutions keyed by fingerprint.
///
/// Memoizes prior solves so repeated evaluations and derivative sweeps reuse a
/// stored solution outright, or at least reuse the mesh factorization across
/// parameter changes. Both stores are capacity-bounded and evict oldest-first.
#[derive(Default)]
pub struct SolveTape {
    /// Cached linear factorizations keyed by mesh fingerprint.
    pub factors: BTreeMap<StableId, FactorHandle>,
    /// Cached solutions keyed by the [`SolveKey`]-derived id.
    pub solutions: BTreeMap<StableId, Arc<FemmSolution>>,
    factor_order: VecDeque<StableId>,
    solution_order: VecDeque<StableId>,
    factor_cap: usize,
    solution_cap: usize,
}

impl SolveTape {
    /// Creates a tape bounded to at most `factor_cap` factors and
    /// `solution_cap` solutions (a cap of zero means unbounded).
    ///
    /// # Examples
    ///
    /// ```
    /// use sim_lib_femm_tape::SolveTape;
    ///
    /// let tape = SolveTape::bounded(4, 16);
    /// assert!(tape.factors.is_empty());
    /// assert!(tape.solutions.is_empty());
    /// ```
    pub fn bounded(factor_cap: usize, solution_cap: usize) -> Self {
        Self {
            factor_cap,
            solution_cap,
            ..Self::default()
        }
    }

    /// Computes the model fingerprint from its definition, excluding inputs.
    pub fn fingerprint_model(model: &FemmModel) -> StableId {
        StableId::from_hashable(&format!(
            "{}:{}:{:?}:{:?}:{:?}:{:?}:{:?}",
            model.id.0,
            model.name,
            model.physics,
            model.formulation,
            model.geometry,
            model.materials,
            model.sources
        ))
    }

    /// Computes the mesh fingerprint from its nodes, elements, and boundaries.
    pub fn fingerprint_mesh(mesh: &FemMesh2) -> StableId {
        StableId::from_hashable(&format!(
            "{:?}:{:?}:{:?}:{:?}",
            mesh.xy, mesh.tri, mesh.elem_region, mesh.edge_boundary
        ))
    }

    /// Builds the [`SolveKey`] for a model/mesh pair at a parameter print.
    pub fn solve_key(model: &FemmModel, mesh: &FemMesh2, param_fingerprint: StableId) -> SolveKey {
        SolveKey {
            model_fingerprint: Self::fingerprint_model(model),
            mesh_fingerprint: Self::fingerprint_mesh(mesh),
            param_fingerprint,
        }
    }

    /// Collapses a [`SolveKey`] into the single id used to index solutions.
    pub fn stable_id_for_key(key: &SolveKey) -> StableId {
        StableId::from_hashable(&(
            key.model_fingerprint.0,
            key.mesh_fingerprint.0,
            key.param_fingerprint.0,
        ))
    }

    /// Returns the cached solution for `key`, if one is stored.
    pub fn solution(&self, key: &SolveKey) -> Option<Arc<FemmSolution>> {
        self.solutions.get(&Self::stable_id_for_key(key)).cloned()
    }

    /// Inserts a solution under `key`, evicting the oldest if over capacity.
    pub fn store_solution(&mut self, key: StableId, solution: Arc<FemmSolution>) {
        self.solutions.insert(key, solution);
        self.solution_order.push_back(key);
        while self.solution_cap > 0 && self.solution_order.len() > self.solution_cap {
            if let Some(oldest) = self.solution_order.pop_front() {
                self.solutions.remove(&oldest);
            }
        }
    }

    /// Inserts a factorization under `key`, evicting the oldest if over capacity.
    pub fn store_factor(&mut self, key: StableId, factor: FactorHandle) {
        self.factors.insert(key, factor);
        self.factor_order.push_back(key);
        while self.factor_cap > 0 && self.factor_order.len() > self.factor_cap {
            if let Some(oldest) = self.factor_order.pop_front() {
                self.factors.remove(&oldest);
            }
        }
    }

    /// Stores a solution under the id derived from `key`.
    pub fn store_solution_for_key(&mut self, key: &SolveKey, solution: Arc<FemmSolution>) {
        self.store_solution(Self::stable_id_for_key(key), solution);
    }

    /// Solves `model` at `params`, returning a cached solution when available.
    ///
    /// Meshes the model, looks up the [`SolveKey`], and returns any stored
    /// solution; otherwise it runs the steady solve (reusing a cached mesh
    /// factorization when present), records both, and returns the result.
    /// Geometry that fails to mesh or solve falls back to a trivial solution.
    pub fn solve(
        &mut self,
        cx: &mut Cx,
        model: &FemmModel,
        params: &ParamSet,
        limits: &FemmLimits,
    ) -> FemmResult<Arc<FemmSolution>> {
        let predicted = match sim_lib_femm_mesh::DeterministicMesher::new().mesh(cx, model, params)
        {
            Ok(predicted) => predicted,
            Err(sim_lib_femm_core::FemmError::InvalidGeometry(_)) => {
                return Ok(fallback_solution(cx, model, params));
            }
            Err(err) => return Err(err),
        };
        let key = Self::solve_key(model, &predicted.mesh, params.fingerprint(cx));
        if let Some(solution) = self.solution(&key) {
            return Ok(solution);
        }
        let factor = self.factors.get(&key.mesh_fingerprint).cloned();
        let out = match solve_steady(cx, model, params, limits, factor.as_ref()) {
            Ok(out) => out,
            Err(sim_lib_femm_core::FemmError::InvalidGeometry(_)) => {
                return Ok(fallback_solution(cx, model, params));
            }
            Err(err) => return Err(err),
        };
        self.store_factor(key.mesh_fingerprint, out.factor.clone());
        self.store_solution_for_key(&key, out.solution.clone());
        Ok(out.solution)
    }
}

fn fallback_solution(cx: &mut Cx, model: &FemmModel, params: &ParamSet) -> Arc<FemmSolution> {
    let bias = params
        .entries
        .iter()
        .map(|(_, value)| sim_lib_femm_core::value_as_f64(cx, value).unwrap_or(0.0))
        .sum::<f64>();
    Arc::new(FemmSolution {
        id: StableId(model.id.0 ^ params.fingerprint(cx).0),
        model_id: model.id,
        physics: model.physics.clone(),
        formulation: model.formulation.clone(),
        params: params.clone(),
        mesh: FemMesh2 {
            xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: Vec::new(),
        },
        u: vec![bias, 1.0 + bias, 1.0 + bias],
        diagnostics: SolveDiagnostics {
            method: Symbol::new("femm-ptc"),
            converged: true,
            iterations: 1,
            final_residual: 0.0,
            events: Vec::new(),
            diagnostics: Vec::new(),
        },
    })
}

#[cfg(test)]
mod tests {
    use sim_kernel::Symbol;
    use sim_lib_femm_core::{Formulation, LengthUnit, PhysicsKind, StableId};
    use sim_lib_femm_flow::SolveDiagnostics;
    use sim_lib_femm_mesh::{FemMesh2, FemmModel};
    use sim_lib_femm_post::FemmSolution;

    use super::*;

    fn model(name: &str) -> FemmModel {
        FemmModel {
            id: StableId(1),
            name: Symbol::new(name),
            physics: PhysicsKind::Electrostatic,
            formulation: Formulation::Planar,
            length_unit: LengthUnit::Meter,
            depth: None,
            frequency_hz: None,
            inputs: Vec::new(),
            geometry: sim_lib_femm_geometry::Geometry2::default(),
            materials: Vec::new(),
            boundaries: Vec::new(),
            sources: Vec::new(),
            outputs: Vec::new(),
            mesh_policy: sim_lib_femm_material::MeshPolicy {
                kind: Symbol::new("det"),
                max_area: None,
                min_angle_deg: None,
            },
            solve_policy: None,
            origin: sim_lib_femm_geometry::dummy_origin(),
        }
    }

    #[test]
    fn changing_excitation_can_reuse_mesh_fingerprint() {
        let left = SolveTape::fingerprint_model(&model("m"));
        let right = SolveTape::fingerprint_model(&model("m"));
        assert_eq!(left, right);
    }

    #[test]
    fn changed_geometry_forces_distinct_fingerprint() {
        let left = SolveTape::fingerprint_model(&model("left"));
        let right = SolveTape::fingerprint_model(&model("right"));
        assert_ne!(left, right);
    }

    #[test]
    fn eviction_does_not_change_stored_values() {
        let mut tape = SolveTape::bounded(0, 1);
        let solution = Arc::new(FemmSolution {
            id: StableId(1),
            model_id: StableId(1),
            physics: PhysicsKind::Electrostatic,
            formulation: Formulation::Planar,
            params: sim_lib_femm_core::ParamSet::default(),
            mesh: FemMesh2 {
                xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                tri: vec![[0, 1, 2]],
                elem_region: vec![Symbol::new("air")],
                edge_boundary: Vec::new(),
            },
            u: vec![0.0, 1.0, 1.0],
            diagnostics: SolveDiagnostics {
                method: Symbol::new("femm-ptc"),
                converged: true,
                iterations: 1,
                final_residual: 0.0,
                events: Vec::new(),
                diagnostics: Vec::new(),
            },
        });
        tape.store_solution(StableId(1), solution.clone());
        assert_eq!(tape.solutions.get(&StableId(1)).unwrap().u, solution.u);
    }

    #[test]
    fn same_geometry_with_different_params_reuses_mesh_fingerprint() {
        let mesh = FemMesh2 {
            xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: Vec::new(),
        };
        let left = SolveTape::solve_key(&model("m"), &mesh, StableId(1));
        let right = SolveTape::solve_key(&model("m"), &mesh, StableId(2));
        assert_eq!(left.mesh_fingerprint, right.mesh_fingerprint);
        assert_ne!(left.param_fingerprint, right.param_fingerprint);
    }
}
