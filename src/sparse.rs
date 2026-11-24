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

/// Single run inside a fixed-size segment/block of states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockRleRun {
    pub state: QState,
    pub count: usize,
}

/// Fixed-size block (segment) of RLE runs for parallel/chunked compression & GPU transfer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RleBlock {
    pub block_len: usize,
    pub runs: Vec<BlockRleRun>,
}

/// Block-segmented Run-Length Encoded representation of a `QIndex`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockRleQIndex {
    pub block_size: usize,
    pub total_len: usize,
    pub blocks: Vec<RleBlock>,
}

impl BlockRleQIndex {
    /// Compresses `index` into independent segments of at most `block_size` states.
    pub fn from_qindex_blocked(index: &QIndex, block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be greater than 0");
        let total_len = index.len();
        if total_len == 0 {
            return Self {
                block_size,
                total_len: 0,
                blocks: Vec::new(),
            };
        }

        let num_blocks = (total_len + block_size - 1) / block_size;
        let mut blocks = Vec::with_capacity(num_blocks);

        let mut iter = index.into_iter();
        for b_idx in 0..num_blocks {
            let remaining = total_len - b_idx * block_size;
            let current_block_len = remaining.min(block_size);

            let mut runs: Vec<BlockRleRun> = Vec::new();
            let mut cur_state: Option<QState> = None;
            let mut cur_count: usize = 0;

            for _ in 0..current_block_len {
                if let Some(state) = iter.next() {
                    match cur_state {
                        Some(s) if s == state => cur_count += 1,
                        Some(s) => {
                            runs.push(BlockRleRun {
                                state: s,
                                count: cur_count,
                            });
                            cur_state = Some(state);
                            cur_count = 1;
                        }
                        None => {
                            cur_state = Some(state);
                            cur_count = 1;
                        }
                    }
                }
            }

            if let Some(s) = cur_state {
                runs.push(BlockRleRun {
                    state: s,
                    count: cur_count,
                });
            }

            blocks.push(RleBlock {
                block_len: current_block_len,
                runs,
            });
        }

        Self {
            block_size,
            total_len,
            blocks,
        }
    }

    /// Decompresses all segments back into a contiguous `QIndex`.
    pub fn to_qindex(&self) -> QIndex {
        let mut out = QIndex::with_capacity(self.total_len);
        for block in &self.blocks {
            for run in &block.runs {
                for _ in 0..run.count {
                    out.push(run.state);
                }
            }
        }
        out
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.total_len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}