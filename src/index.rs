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