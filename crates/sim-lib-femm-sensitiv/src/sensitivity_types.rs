use std::sync::Arc;

use sim_lib_femm_mesh::FemMesh2;
use sim_lib_femm_post::FemmSolution;
use sim_lib_numbers_ad::Dual;

pub(crate) struct DiffMesh {
    pub mesh: FemMesh2,
    pub dxy: Vec<[f64; 2]>,
}

pub(crate) struct DiffSolution {
    pub solution: Arc<FemmSolution>,
    pub du: Vec<f64>,
    pub dxy: Vec<[f64; 2]>,
}

pub(crate) struct DualGeom {
    pub xy: [[Dual<1>; 2]; 3],
    pub area: Dual<1>,
    pub grad: [[Dual<1>; 2]; 3],
}
