// ============================================================================
// UNSAFE & ITERATOR SAFETY OVERVIEW (iter.rs)
// ============================================================================
//
// 1. `QIndexIter::next` — `self.slice.get_unchecked(u32_idx)`:
//    Reads a packed `u32` word without bounds checking after verifying
//    `self.current < self.len`[cite: 49]. Since `u32_idx = self.current / 16`,
//    the index always falls within the backing slice backing `self.len` states[cite: 49].
//
// 2. `QIndexIter::next` — `QState::from_u8_unchecked(val)`:
//    Masks the extracted 2-bit state with `0b11` (`val` in `0..=3`), ensuring
//    the bit pattern is always a valid `QState` discriminant[cite: 49].
//
// 3. `QIndexIterMut::next` — `self.slice.get_unchecked(u32_idx)`:
//    Accesses the packed `u32` word at `idx / 16` after confirming
//    `self.current < self.len` (`idx = self.current`), keeping the read within
//    the initialized slice bounds[cite: 50].
//
// 4. `QIndexIterMut::next` — `QState::from_u8_unchecked(val)`:
//    Converts `((chunk >> bit_shift) & 0b11) as u8` into `QState`[cite: 50].
//    Because `& 0b11` restricts `val` to `0..=3`, constructing `QState` is
//    guaranteed valid[cite: 50].
// ============================================================================

use crate::index::QIndex;
use crate::state::QState;

/// Immutable iterator over packed `QState` items in `QIndex`.
pub struct QIndexIter<'a> {
    slice: &'a [u32],
    len: usize,
    current: usize,
}

impl<'a> QIndexIter<'a> {
    #[inline]
    pub fn new(index: &'a QIndex) -> Self {
        Self {
            slice: index.as_raw_slice(),
            len: index.len(),
            current: 0,
        }
    }
}

impl<'a> Iterator for QIndexIter<'a> {
    type Item = QState;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.len {
            return None;
        }

        let u32_idx = self.current / 16;
        let bit_shift = (self.current % 16) * 2;

        // Safe: self.current < self.len bounds-checked above
        let chunk = unsafe { *self.slice.get_unchecked(u32_idx) };
        let val = ((chunk >> bit_shift) & 0b11) as u8;

        self.current += 1;
        unsafe { Some(QState::from_u8_unchecked(val)) }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for QIndexIter<'a> {}

impl<'a> IntoIterator for &'a QIndex {
    type Item = QState;
    type IntoIter = QIndexIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        QIndexIter::new(self)
    }
}

pub struct QIndexIterMut<'a> {
    slice: &'a mut [u32],
    len: usize,
    current: usize,
}

impl<'a> QIndexIterMut<'a> {
    #[inline]
    pub fn new(index: &'a mut QIndex) -> Self {
        let len = index.len(); // 1. Prvo uzmemo dužinu (pozajmica se odmah završava)
        Self {
            slice: index.as_mut_raw_slice(), // 2. Zatim uzmemo mutabilnu pozajmicu koja traje 'a
            len,
            current: 0,
        }
    }
}

impl<'a> Iterator for QIndexIterMut<'a> {
    type Item = (usize, QState); // Returns index and state

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.len {
            return None;
        }
        let idx = self.current;
        let u32_idx = idx / 16;
        let bit_shift = (idx % 16) * 2;

        let chunk = unsafe { *self.slice.get_unchecked(u32_idx) };
        let val = ((chunk >> bit_shift) & 0b11) as u8;

        self.current += 1;
        let state = unsafe { QState::from_u8_unchecked(val) };
        Some((idx, state))
    }
}

/// Represents a single chunk slice ready for streaming or staging transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferChunk<'a> {
    pub chunk_index: usize,
    pub state_offset: usize,
    pub state_count: usize,
    pub raw_u32_words: &'a [u32],
}

/// Streaming iterator over a `QIndex` split into fixed-size quaternary state chunks.
pub struct QuatChunkStream<'a> {
    index: &'a QIndex,
    states_per_chunk: usize,
    next_state_offset: usize,
    next_chunk_index: usize,
}

impl<'a> QuatChunkStream<'a> {
    /// Creates a new chunk stream where each chunk covers up to `states_per_chunk` states.
    /// Rounds down to multiples of 16 when possible so `u32` word boundaries stay aligned.
    pub fn new(index: &'a QIndex, states_per_chunk: usize) -> Self {
        let aligned_chunk = if states_per_chunk >= 16 {
            (states_per_chunk / 16) * 16
        } else {
            states_per_chunk.max(1)
        };

        Self {
            index,
            states_per_chunk: aligned_chunk,
            next_state_offset: 0,
            next_chunk_index: 0,
        }
    }

    /// Total number of chunks in this stream.
    pub fn total_chunks(&self) -> usize {
        if self.index.len() == 0 {
            0
        } else {
            (self.index.len() + self.states_per_chunk - 1) / self.states_per_chunk
        }
    }
}

impl<'a> Iterator for QuatChunkStream<'a> {
    type Item = TransferChunk<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let total_states = self.index.len();
        if self.next_state_offset >= total_states {
            return None;
        }

        let remaining_states = total_states - self.next_state_offset;
        let state_count = remaining_states.min(self.states_per_chunk);
        let state_offset = self.next_state_offset;
        let chunk_index = self.next_chunk_index;

        let word_start = state_offset / 16;
        let word_end = (state_offset + state_count + 15) / 16;
        let full_slice = self.index.as_raw_slice();
        let raw_u32_words = &full_slice[word_start..word_end.min(full_slice.len())];

        self.next_state_offset += state_count;
        self.next_chunk_index += 1;

        Some(TransferChunk {
            chunk_index,
            state_offset,
            state_count,
            raw_u32_words,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_chunks().saturating_sub(self.next_chunk_index);
        (remaining, Some(remaining))
    }
}

/// Streams a `QIndex` in chunks and invokes `consumer` for each chunk.
pub fn stream_qindex_chunks<F>(index: &QIndex, states_per_chunk: usize, mut consumer: F)
where
    F: FnMut(TransferChunk<'_>),
{
    for chunk in QuatChunkStream::new(index, states_per_chunk) {
        consumer(chunk);
    }
}