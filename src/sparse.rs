use crate::state::QState;
use crate::index::QIndex;
use serde::{Serialize, Deserialize};

/// Represents a single run of identical quaternary states in RLE compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RleRun {
    pub state: QState,
    pub count: usize,
}

/// A compressed, sparse representation of a QIndex using Run-Length Encoding (RLE).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RleQIndex {
    runs: Vec<RleRun>,
    total_len: usize,
}

impl RleQIndex {
    /// Compresses a standard QIndex into an RLE sparse structure.
    pub fn from_qindex(index: &QIndex) -> Self {
        let mut runs = Vec::new();
        if index.len() == 0 {
            return Self { runs, total_len: 0 };
        }

        let mut current_state: Option<QState> = None;
        let mut current_count = 0;

        for state in index.into_iter() {
            if let Some(s) = current_state {
                if s == state {
                    current_count += 1;
                } else {
                    runs.push(RleRun { state: s, count: current_count });
                    current_state = Some(state);
                    current_count = 1;
                }
            } else {
                current_state = Some(state);
                current_count = 1;
            }
        }

        if let Some(s) = current_state {
            runs.push(RleRun { state: s, count: current_count });
        }

        Self {
            runs,
            total_len: index.len(),
        }
    }

    /// Decompresses the RLE structure back into a standard QIndex.
    pub fn to_qindex(&self) -> QIndex {
        let mut index = QIndex::with_capacity(self.total_len);
        for run in &self.runs {
            for _ in 0..run.count {
                index.push(run.state);
            }
        }
        index
    }

    /// Returns the total number of states in the compressed index.
    #[inline]
    pub fn len(&self) -> usize {
        self.total_len
    }

    /// Checks if the compressed index is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    /// Returns a reference to the internal RLE runs slice.
    #[inline]
    pub fn runs(&self) -> &[RleRun] {
        &self.runs
    }
}