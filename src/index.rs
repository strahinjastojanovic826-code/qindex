// ============================================================================
// UNSAFE & MEMORY SAFETY OVERVIEW (index.rs)
// ============================================================================
//
// 1. `unsafe impl Send / Sync for GpuBufferInfo`:
//    Allows transferring and sharing `GpuBufferInfo` descriptors across threads[cite: 40].
//    The caller must ensure the underlying buffer pointed to by `ptr` remains
//    valid and properly synchronized while referenced across threads[cite: 40].
//
// 2. `try_with_capacity_and_alignment`:
//    Allocates `u32` storage via `alloc(layout)` and zero-initializes it with
//    `std::ptr::write_bytes`[cite: 41]. `NonNull::new` verifies allocation
//    success, and `layout` matches `capacity_u32` and `align`[cite: 41].
//
// 3. `set` & `push`:
//    Compute `u32_idx` and `bit_shift` for 2-bit packed quaternary states and
//    update the target `u32` lane via pointer arithmetic (`ptr.add(u32_idx)`)[cite: 41, 42].
//    `set` asserts `index < self.len`, and `push` ensures capacity via
//    `reserve(1)` before writing[cite: 41, 42].
//
// 4. `as_mut_raw_slice`, `as_raw_slice`, & `as_bytes`:
//    Construct Rust slices (`&mut [u32]`, `&[u32]`, `&[u8]`) from `self.ptr`
//    and computed lengths[cite: 42, 44]. Lifetimes are tied to `&mut self` or
//    `&self`, and the buffer is initialized up to `capacity_u32` / `byte_size`[cite: 42, 44].
//
// 5. `clear`:
//    Zeroes only the active `u32_used` words (`(self.len + 15) / 16`) with
//    `write_bytes` before resetting `self.len = 0`[cite: 42].
//
// 6. `push_u32_chunk`:
//    Writes a 16-state packed `u32` chunk at `u32_idx` and `bit_shift`,
//    splitting across two adjacent `u32` words when `bit_shift != 0` after
//    ensuring capacity for 16 states[cite: 43].
//
// 7. `reserve`:
//    Reallocates the backing buffer via `realloc` from `old_layout` to
//    `new_layout.size()` and zero-initializes newly added `u32` words[cite: 43].
//
// 8. `get`:
//    Reads the `u32` word at `u32_idx`, shifts by `bit_shift`, masks `0b11`,
//    and converts via `QState::from_u8_unchecked` after bounds checking
//    `index < self.len`[cite: 44].
//
// 9. `vec_not`, `vec_and`, `vec_or`, & `vec_xor`:
//    Iterate over `u32_used` packed `u32` lanes and write results into a
//    newly allocated `QIndex` of matching length and alignment[cite: 45, 46].
//    Source and destination buffers are distinct allocations with valid
//    lengths.
//
// 10. `Clone::clone`:
//     Copies `u32_used` words from `self.ptr` to `new_index.ptr` using
//     `std::ptr::copy_nonoverlapping` into a separate allocation[cite: 47].
//
// 11. `Drop::drop` & `unsafe impl Send / Sync for QIndex`:
//     Reconstructs the `Layout` from `(self.capacity_u32, self.align)` and
//     frees the buffer with `dealloc` when `QIndex` is dropped[cite: 48].
//     `Send` and `Sync` are implemented for `QIndex` since it owns its
//     aligned allocation and exposes safe read/write APIs[cite: 48].
// ============================================================================

use std::alloc::{alloc, dealloc, realloc, Layout, LayoutError};
use std::fmt;
use std::ptr::NonNull;
use crate::state::QState;

/// Cache-line & GPU transfer aligned memory layout constraint (default 64 bytes).
pub const DEFAULT_ALIGNMENT: usize = 64;

/// Packed 2-bit quantum state buffer optimized for CPU execution and zero-copy GPU transfers.
pub struct QIndex {
    ptr: NonNull<u32>,
    capacity_u32: usize,
    len: usize,
    align: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuBufferInfo {
    pub ptr: *const std::ffi::c_void,
    pub byte_size: usize,
    pub capacity_u32: usize,
    pub alignment: usize,
}

unsafe impl Send for GpuBufferInfo {}
unsafe impl Sync for GpuBufferInfo {}

#[inline]
fn u32_min_quat(a: u32, b: u32) -> u32 {
    // Compare each 2-bit lane (0..=3) independently using borrow bit at MSB of each lane.
    let even_mask: u32 = 0x5555_5555;
    let odd_mask: u32 = 0xAAAA_AAAA;

    // For each 2-bit pair (hi:lo), borrow when a < b:
    // (~a & b) has 1 where a=0, b=1.
    let a_even = a & even_mask;
    let b_even = b & even_mask;
    let a_odd = (a & odd_mask) >> 1;
    let b_odd = (b & odd_mask) >> 1;

    // 1 where lane a < lane b
    let lt_lo = (!a_even) & b_even;
    let eq_lo = !(a_even ^ b_even) & even_mask;
    let lt_hi = ((!a_odd) & b_odd) | (eq_lo & ((!a_odd) & b_odd));

    let lt_any = (lt_hi << 1) | lt_lo;
    // Select b where lt_any lane is non-zero, otherwise a
    let sel_b = ((lt_any | (lt_any >> 1)) & even_mask) * 3;
    (a & !sel_b) | (b & sel_b)
}

#[inline]
fn u32_max_quat(a: u32, b: u32) -> u32 {
    let even_mask: u32 = 0x5555_5555;
    let odd_mask: u32 = 0xAAAA_AAAA;

    let a_even = a & even_mask;
    let b_even = b & even_mask;
    let a_odd = (a & odd_mask) >> 1;
    let b_odd = (b & odd_mask) >> 1;

    // 1 where lane a > lane b
    let gt_lo = a_even & (!b_even);
    let eq_lo = !(a_even ^ b_even) & even_mask;
    let gt_hi = (a_odd & (!b_odd)) | (eq_lo & (a_odd & (!b_odd)));

    let gt_any = (gt_hi << 1) | gt_lo;
    let sel_b = ((gt_any | (gt_any >> 1)) & even_mask) * 3;
    (a & !sel_b) | (b & sel_b)
}

impl QIndex {
    #[inline]
    fn layout_for(capacity_u32: usize, align: usize) -> Result<Layout, LayoutError> {
        let bytes = capacity_u32.checked_mul(std::mem::size_of::<u32>()).ok_or_else(|| {
            // Force LayoutError via invalid alignment if overflow occurs
            Layout::from_size_align(usize::MAX, 1).unwrap_err()
        })?;
        Layout::from_size_align(bytes, align)
    }

    /// Constructs a new empty `QIndex` with default initial capacity.
    pub fn new() -> Self {
        Self::with_capacity(64)
    }

    /// Returns a GPU-compatible buffer descriptor for safe zero-copy interop.
    #[inline]
    pub fn gpu_buffer_info(&self) -> GpuBufferInfo {
        GpuBufferInfo {
            ptr: self.ptr.as_ptr() as *const std::ffi::c_void,
            byte_size: self.byte_size(),
            capacity_u32: self.capacity_u32,
            alignment: self.align,
        }
    }

    /// Creates a `QIndex` with custom capacity and default 64-byte alignment.
    pub fn with_capacity(capacity_states: usize) -> Self {
        Self::with_capacity_and_alignment(capacity_states, DEFAULT_ALIGNMENT)
    }

    /// Creates a `QIndex` specifying both state capacity and custom alignment (e.g. 256 for Vulkan/WebGPU).
    pub fn with_capacity_and_alignment(capacity_states: usize, align: usize) -> Self {
        Self::try_with_capacity_and_alignment(capacity_states, align)
            .expect("Failed to allocate memory for QIndex")
    }

    /// Fallible constructor returning `Result` instead of unwrapping/panicking.
    pub fn try_with_capacity_and_alignment(
        capacity_states: usize,
        align: usize,
    ) -> Result<Self, &'static str> {
        let u32_needed = (capacity_states + 15) / 16;
        let capacity_u32 = u32_needed.max(4); // Min 16 bytes allocation

        let layout = Self::layout_for(capacity_u32, align).map_err(|_| "Invalid memory layout")?;

        let raw_ptr = unsafe { alloc(layout) as *mut u32 };
        let ptr = NonNull::new(raw_ptr).ok_or("Allocation failed")?;

        // Zero-initialize allocated memory
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0, capacity_u32);
        }

        Ok(Self {
            ptr,
            capacity_u32,
            len: 0,
            align,
        })
    }

    #[inline]
pub fn set(&mut self, index: usize, state: QState) {
    assert!(index < self.len, "Index out of bounds");
    let u32_idx = index / 16;
    let bit_shift = (index % 16) * 2;

    unsafe {
        let elem_ptr = self.ptr.as_ptr().add(u32_idx);
        let current_val = *elem_ptr;
        let mask = !(0b11u32 << bit_shift);
        *elem_ptr = (current_val & mask) | ((state as u32) << bit_shift);
    }
}

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity_u32 * 16
    }

    #[inline]
    pub fn byte_size(&self) -> usize {
        (self.len + 15) / 4
    }

    #[inline]
pub fn as_mut_raw_slice(&mut self) -> &mut [u32] {
    unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.capacity_u32) }
}

    /// Resets the container without deallocating backing storage.
    pub fn clear(&mut self) {
        let u32_used = (self.len + 15) / 16;
        if u32_used > 0 {
            unsafe {
                std::ptr::write_bytes(self.ptr.as_ptr(), 0, u32_used);
            }
        }
        self.len = 0;
    }

    /// Pushes a single 2-bit `QState` into the index.
    #[inline]
    pub fn push(&mut self, state: QState) {
        if self.len >= self.capacity_u32 * 16 {
            self.reserve(1);
        }

        let u32_idx = self.len / 16;
        let bit_shift = (self.len % 16) * 2;

        unsafe {
            let elem_ptr = self.ptr.as_ptr().add(u32_idx);
            let current_val = *elem_ptr;
            let mask = !(0b11u32 << bit_shift);
            *elem_ptr = (current_val & mask) | ((state as u32) << bit_shift);
        }

        self.len += 1;
    }

    /// Appends a chunk of 16 states packed into a single `u32`.
    pub fn push_u32_chunk(&mut self, chunk: u32) {
        let u32_idx = self.len / 16;
        let bit_shift = (self.len % 16) * 2;

        if self.len + 16 > self.capacity_u32 * 16 {
            self.reserve(16);
        }

        unsafe {
            let elem_ptr = self.ptr.as_ptr().add(u32_idx);
            if bit_shift == 0 {
                *elem_ptr = chunk;
            } else {
                let overflow_bits = 32 - bit_shift;
                *elem_ptr |= chunk << bit_shift;
                *elem_ptr.add(1) = chunk >> overflow_bits;
            }
        }

        self.len += 16;
    }

    /// Ensures capacity for at least `additional` more states.
    pub fn reserve(&mut self, additional: usize) {
        let needed_states = self.len + additional;
        let needed_u32 = (needed_states + 15) / 16;

        if needed_u32 <= self.capacity_u32 {
            return;
        }

        let mut new_capacity_u32 = self.capacity_u32 * 2;
        while new_capacity_u32 < needed_u32 {
            new_capacity_u32 *= 2;
        }

        let old_layout = Self::layout_for(self.capacity_u32, self.align)
            .expect("Valid layout calculation failed");
        let new_layout = Self::layout_for(new_capacity_u32, self.align)
            .expect("Valid layout calculation failed");

        unsafe {
            let new_raw_ptr = realloc(
                self.ptr.as_ptr() as *mut u8,
                old_layout,
                new_layout.size(),
            ) as *mut u32;

            let new_ptr = NonNull::new(new_raw_ptr).expect("Reallocation failed");

            // Zero-initialize newly allocated storage block
            let added_u32 = new_capacity_u32 - self.capacity_u32;
            std::ptr::write_bytes(new_ptr.as_ptr().add(self.capacity_u32), 0, added_u32);

            self.ptr = new_ptr;
            self.capacity_u32 = new_capacity_u32;
        }
    }

    /// Gets state at a specific index, returning `None` if out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<QState> {
        if index >= self.len {
            return None;
        }

        let u32_idx = index / 16;
        let bit_shift = (index % 16) * 2;

        unsafe {
            let val = (*self.ptr.as_ptr().add(u32_idx)) >> bit_shift;
            Some(QState::from_u8_unchecked((val & 0b11) as u8))
        }
    }

    pub fn count_state(&self, target: QState) -> usize {
    let target_val = target as u32;
    // Create a pattern for the target state across all 16 positions in a u32
    let mut pattern = 0u32;
    for i in 0..16 {
        pattern |= target_val << (i * 2);
    }
    let mask = 0xFFFF_FFFF; // all 16 quats checked
    self.count_matches(pattern, mask)
}

    /// Exposes internal buffer as raw `&[u32]` slice.
    #[inline]
    pub fn as_raw_slice(&self) -> &[u32] {
        let u32_used = (self.len + 15) / 16;
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), u32_used) }
    }

    /// Exposes internal raw bytes slice for Direct-to-GPU memory transfer (Zero-Copy).
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const u8, self.byte_size()) }
    }

    pub fn par_count_matches(&self, pattern: u32, mask: u32) -> usize {
    use rayon::prelude::*;
    let slice = self.as_raw_slice();
    
    slice.par_iter()
        .map(|&chunk| {
            let matched = !(chunk ^ pattern) & mask;
            let even_bits = matched & 0x5555_5555;
            let odd_bits = (matched >> 1) & 0x5555_5555;
            (even_bits & odd_bits).count_ones() as usize
        })
        .sum()
}

    /// Bulk appends states from a slice.
    pub fn extend_from_slice(&mut self, states: &[QState]) {
        self.reserve(states.len());
        for &state in states {
            self.push(state);
        }
    }

    /// Counts matches of pattern matching target bitmask across packed storage.
    pub fn count_matches(&self, pattern: u32, mask: u32) -> usize {
        let slice = self.as_raw_slice();
        let mut total_matches = 0;

        for &chunk in slice {
            let matched = !(chunk ^ pattern) & mask;
            let even_bits = matched & 0x5555_5555;
            let odd_bits = (matched >> 1) & 0x5555_5555;
            let valid_quats = even_bits & odd_bits;

            total_matches += valid_quats.count_ones() as usize;
        }

        total_matches
    }

    /// Vector quaternary NOT across all states in the index.
    pub fn vec_not(&self) -> Self {
        let mut out = Self::with_capacity_and_alignment(self.len, self.align);
        let u32_used = (self.len + 15) / 16;
        if u32_used > 0 {
            unsafe {
                let src = self.ptr.as_ptr();
                let dst = out.ptr.as_ptr();
                for i in 0..u32_used {
                    // Invert all 2-bit lanes (0xFFFF_FFFF flips every bit)
                    *dst.add(i) = !(*src.add(i));
                }
            }
        }
        out.len = self.len;
        out
    }

    /// Vector quaternary AND (element-wise min) between two indices of equal length.
    pub fn vec_and(&self, other: &Self) -> Self {
        assert_eq!(self.len, other.len, "QIndex length mismatch in vec_and");
        let mut out = Self::with_capacity_and_alignment(self.len, self.align);
        let u32_used = (self.len + 15) / 16;
        if u32_used > 0 {
            unsafe {
                let a_ptr = self.ptr.as_ptr();
                let b_ptr = other.ptr.as_ptr();
                let dst = out.ptr.as_ptr();
                for i in 0..u32_used {
                    *dst.add(i) = u32_min_quat(*a_ptr.add(i), *b_ptr.add(i));
                }
            }
        }
        out.len = self.len;
        out
    }

    /// Vector quaternary OR (element-wise max) between two indices of equal length.
    pub fn vec_or(&self, other: &Self) -> Self {
        assert_eq!(self.len, other.len, "QIndex length mismatch in vec_or");
        let mut out = Self::with_capacity_and_alignment(self.len, self.align);
        let u32_used = (self.len + 15) / 16;
        if u32_used > 0 {
            unsafe {
                let a_ptr = self.ptr.as_ptr();
                let b_ptr = other.ptr.as_ptr();
                let dst = out.ptr.as_ptr();
                for i in 0..u32_used {
                    *dst.add(i) = u32_max_quat(*a_ptr.add(i), *b_ptr.add(i));
                }
            }
        }
        out.len = self.len;
        out
    }

    /// Vector quaternary XOR between two indices of equal length.
    pub fn vec_xor(&self, other: &Self) -> Self {
        assert_eq!(self.len, other.len, "QIndex length mismatch in vec_xor");
        let mut out = Self::with_capacity_and_alignment(self.len, self.align);
        let u32_used = (self.len + 15) / 16;
        if u32_used > 0 {
            unsafe {
                let a_ptr = self.ptr.as_ptr();
                let b_ptr = other.ptr.as_ptr();
                let dst = out.ptr.as_ptr();
                for i in 0..u32_used {
                    *dst.add(i) = (*a_ptr.add(i)) ^ (*b_ptr.add(i));
                }
            }
        }
        out.len = self.len;
        out
    }

    /// Finds the first index containing `target` state, or `None` if not found.
    pub fn find_first(&self, target: QState) -> Option<usize> {
        let target_u8 = target as u8;
        let slice = self.as_raw_slice();
        for (u32_idx, &chunk) in slice.iter().enumerate() {
            let base = u32_idx * 16;
            for lane in 0..16 {
                let idx = base + lane;
                if idx >= self.len {
                    return None;
                }
                let val = ((chunk >> (lane * 2)) & 0b11) as u8;
                if val == target_u8 {
                    return Some(idx);
                }
            }
        }
        None
    }

    /// Returns all state indices matching `target`.
    pub fn positions(&self, target: QState) -> Vec<usize> {
        let target_u8 = target as u8;
        let mut result = Vec::new();
        let slice = self.as_raw_slice();
        for (u32_idx, &chunk) in slice.iter().enumerate() {
            let base = u32_idx * 16;
            for lane in 0..16 {
                let idx = base + lane;
                if idx >= self.len {
                    return result;
                }
                let val = ((chunk >> (lane * 2)) & 0b11) as u8;
                if val == target_u8 {
                    result.push(idx);
                }
            }
        }
        result
    }

    /// Filters and returns a new `QIndex` keeping only states equal to `target`.
    pub fn filter_state(&self, target: QState) -> Self {
        let target_u8 = target as u8;
        let mut filtered = Self::with_capacity_and_alignment(self.len, self.align);
        let slice = self.as_raw_slice();
        for (u32_idx, &chunk) in slice.iter().enumerate() {
            let base = u32_idx * 16;
            for lane in 0..16 {
                if base + lane >= self.len {
                    return filtered;
                }
                let val = ((chunk >> (lane * 2)) & 0b11) as u8;
                if val == target_u8 {
                    filtered.push(target);
                }
            }
        }
        filtered
    }

    /// Filters and returns a new `QIndex` keeping only states matching `predicate`.
    pub fn filter_by<F>(&self, mut predicate: F) -> Self
    where
        F: FnMut(QState) -> bool,
    {
        let mut filtered = Self::with_capacity_and_alignment(self.len, self.align);
        for state in self.into_iter() {
            if predicate(state) {
                filtered.push(state);
            }
        }
        filtered
    }

}

impl Default for QIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for QIndex {
    fn clone(&self) -> Self {
        let mut new_index = Self::with_capacity_and_alignment(self.len, self.align);
        let u32_used = (self.len + 15) / 16;

        if u32_used > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_index.ptr.as_ptr(), u32_used);
            }
        }
        new_index.len = self.len;
        new_index
    }
}

impl PartialEq for QIndex {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        self.as_raw_slice() == other.as_raw_slice()
    }
}

impl Eq for QIndex {}

impl Extend<QState> for QIndex {
    fn extend<T: IntoIterator<Item = QState>>(&mut self, iter: T) {
        let iterator = iter.into_iter();
        let (lower_bound, _) = iterator.size_hint();
        if lower_bound > 0 {
            self.reserve(lower_bound);
        }
        for state in iterator {
            self.push(state);
        }
    }
}

impl FromIterator<QState> for QIndex {
    fn from_iter<T: IntoIterator<Item = QState>>(iter: T) -> Self {
        let mut index = QIndex::new();
        index.extend(iter);
        index
    }
}

impl Drop for QIndex {
    fn drop(&mut self) {
        let layout = Self::layout_for(self.capacity_u32, self.align)
            .expect("Valid layout calculation during drop failed");
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

unsafe impl Send for QIndex {}
unsafe impl Sync for QIndex {}

impl fmt::Debug for QIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QIndex")
            .field("len", &self.len)
            .field("byte_size", &self.byte_size())
            .field("capacity_u32", &self.capacity_u32)
            .field("align", &self.align)
            .finish()
    }
}

impl serde::Serialize for QIndex {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(self.len))?;
        for state in self.into_iter() {
            seq.serialize_element(&state)?;
        }
        seq.end()
    }
}

impl<'de> serde::Deserialize<'de> for QIndex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct QIndexVisitor;
        impl<'de> serde::de::Visitor<'de> for QIndexVisitor {
            type Value = QIndex;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a sequence of QState values")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut index = QIndex::new();
                while let Some(state) = seq.next_element()? {
                    index.push(state);
                }
                Ok(index)
            }
        }
        deserializer.deserialize_seq(QIndexVisitor)
    }
}