use crate::{FemmError, FemmResult, implementation::StableId};

/// A square sparse matrix in compressed-sparse-row (CSR) form.
///
/// The shared assembled-system representation for FEMM linear solves; carries
/// `f64` entries and validates its own structural invariants.
///
/// # Examples
///
/// ```
/// use sim_lib_femm_core::CsrMatrix;
///
/// let identity = CsrMatrix::identity(3);
/// assert_eq!(identity.rows(), 3);
/// assert_eq!(identity.matvec(&[1.0, 2.0, 3.0]).unwrap(), vec![1.0, 2.0, 3.0]);
///
/// // A malformed structure is rejected at construction time.
/// assert!(CsrMatrix::new(vec![0, 1], vec![3], vec![1.0]).is_err());
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct CsrMatrix {
    /// Row pointers of length `rows + 1`, starting at zero and monotone.
    pub rowptr: Vec<usize>,
    /// Column index for each stored nonzero.
    pub colind: Vec<usize>,
    /// Value for each stored nonzero, parallel to [`Self::colind`].
    pub vals: Vec<f64>,
}

impl CsrMatrix {
    /// Build a CSR matrix from raw arrays, validating its structure.
    ///
    /// Returns [`FemmError::MalformedMatrix`] if the arrays are inconsistent.
    pub fn new(rowptr: Vec<usize>, colind: Vec<usize>, vals: Vec<f64>) -> FemmResult<Self> {
        let matrix = Self {
            rowptr,
            colind,
            vals,
        };
        matrix.validate()?;
        Ok(matrix)
    }

    /// Construct the `n` by `n` identity matrix in CSR form.
    pub fn identity(n: usize) -> Self {
        Self {
            rowptr: (0..=n).collect(),
            colind: (0..n).collect(),
            vals: vec![1.0; n],
        }
    }

    /// Number of rows (equivalently columns) in the matrix.
    pub fn rows(&self) -> usize {
        self.rowptr.len().saturating_sub(1)
    }

    /// Check the CSR structural invariants, returning [`FemmError::MalformedMatrix`] on failure.
    pub fn validate(&self) -> FemmResult<()> {
        if self.rowptr.is_empty() || self.rowptr[0] != 0 {
            return Err(FemmError::MalformedMatrix(
                "rowptr must start at zero".to_owned(),
            ));
        }
        if self.rowptr.windows(2).any(|w| w[0] > w[1]) {
            return Err(FemmError::MalformedMatrix(
                "rowptr must be monotone".to_owned(),
            ));
        }
        let Some(last) = self.rowptr.last().copied() else {
            return Err(FemmError::MalformedMatrix("missing rowptr".to_owned()));
        };
        if last != self.colind.len() || last != self.vals.len() {
            return Err(FemmError::MalformedMatrix(
                "rowptr tail must equal nnz".to_owned(),
            ));
        }
        let rows = self.rows();
        if self.colind.iter().any(|&index| index >= rows) {
            return Err(FemmError::MalformedMatrix(
                "column index out of range".to_owned(),
            ));
        }
        if self.vals.iter().any(|value| !value.is_finite()) {
            return Err(FemmError::MalformedMatrix(
                "matrix values must be finite".to_owned(),
            ));
        }
        Ok(())
    }

    /// Multiply the matrix by dense vector `x`, returning the dense result.
    pub fn matvec(&self, x: &[f64]) -> FemmResult<Vec<f64>> {
        self.validate()?;
        let rows = self.rows();
        if x.len() != rows {
            return Err(FemmError::MalformedMatrix(format!(
                "matvec vector length {} does not match matrix rows {rows}",
                x.len()
            )));
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(FemmError::MalformedMatrix(
                "matvec vector values must be finite".to_owned(),
            ));
        }
        let mut out = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = self.rowptr[row];
            let end = self.rowptr[row + 1];
            let mut sum = 0.0;
            for idx in start..end {
                sum += self.vals[idx] * x[self.colind[idx]];
                if !sum.is_finite() {
                    return Err(FemmError::MalformedMatrix(
                        "matvec produced a non-finite value".to_owned(),
                    ));
                }
            }
            out.push(sum);
        }
        Ok(out)
    }

    /// Expand the matrix into a dense row-major `Vec<Vec<f64>>`.
    pub fn to_dense(&self) -> FemmResult<Vec<Vec<f64>>> {
        self.validate()?;
        let n = self.rows();
        let mut dense = vec![vec![0.0; n]; n];
        for (row, dense_row) in dense.iter_mut().enumerate().take(n) {
            for idx in self.rowptr[row]..self.rowptr[row + 1] {
                let value = dense_row[self.colind[idx]] + self.vals[idx];
                if !value.is_finite() {
                    return Err(FemmError::MalformedMatrix(
                        "dense expansion produced a non-finite value".to_owned(),
                    ));
                }
                dense_row[self.colind[idx]] = value;
            }
        }
        Ok(dense)
    }

    /// Compute a content [`StableId`] over the matrix structure and values.
    pub fn fingerprint(&self) -> StableId {
        let text = format!("{:?}{:?}{:?}", self.rowptr, self.colind, self.vals);
        StableId::from_hashable(&text)
    }
}
