// ============================================================================
// UNSAFE & FFI SAFETY OVERVIEW (ffi.rs)
// ============================================================================
//
// 1. `qindex_new` & `qindex_with_capacity_and_alignment`:
//    Allocates a `QIndex` on the Rust heap and returns a raw `*mut QIndex`
//    via `Box::into_raw`[cite: 34]. The caller takes ownership and must release
//    the handle via `qindex_free` without double-freeing or aliasing[cite: 34].
//
// 2. `qindex_push`:
//    Checks for null pointer and invalid state (`> 3`), then dereferences `ptr`
//    as `&mut *ptr` inside `unsafe`[cite: 34]. The caller must guarantee `ptr`
//    points to a live, properly aligned `QIndex` with exclusive mutable access.
//
// 3. `qindex_len`:
//    Checks `ptr.is_null()` and dereferences `(*ptr).len()` inside `unsafe`[cite: 35].
//    `ptr` must reference a valid, initialized `QIndex` instance.
//
// 4. `qindex_as_raw_slice`:
//    Dereferences `ptr` (`&*ptr`), writes `out_len` when non-null, and returns
//    `slice.as_ptr()` (`*const u32`)[cite: 35]. The returned pointer remains valid
//    only while the underlying `QIndex` is neither mutated nor freed.
//
// 5. `qindex_free`:
//    When `!ptr.is_null()`, reconstructs and drops the `Box<QIndex>` via
//    `drop(Box::from_raw(ptr))`[cite: 35]. Must be called once per allocation
//    using a pointer created by `qindex_new` or
//    `qindex_with_capacity_and_alignment`[cite: 34, 35].
//
// 6. `QIndexAllocator::deallocate` & `DefaultQIndexAllocator`:
//    Uses `unsafe { alloc(layout) }` and `dealloc(ptr, layout)`[cite: 36]. Callers
//    must pass matching `size`, `align`, and `GpuMemoryProfile` values and
//    ensure non-null pointers are only freed once[cite: 36, 37].
//
// 7. `qindex_alloc_staging` & `qindex_free_staging`:
//    Allocates and releases aligned `StagedTransferBuffer` memory for
//    host/GPU staging[cite: 37]. `qindex_free_staging` must be called at most
//    once per non-null buffer with its original `byte_len`, `align`, and
//    `profile`[cite: 37].
//
// 8. `unsafe impl Send / Sync for StagedTransferBuffer`:
//    Marks `StagedTransferBuffer` as safe to transfer or share across threads[cite: 38].
//    Callers must still coordinate lifetimes and synchronization around the
//    underlying raw pointer across threads[cite: 38].
//
// 9. `validate_qindex_const_ptr` & `validate_qindex_mut_ptr`:
//    Checks for null and alignment before forming `&'static QIndex` or
//    `&'static mut QIndex`[cite: 38]. Callers must ensure no overlapping mutable
//    references or dangling pointers exist.
//
// 10. `qindex_push_checked` & `qindex_get_checked`:
//     Validates `state` (`0..=3`), pointer validity/alignment, and bounds
//     before mutating `QIndex` or writing `*out_state` (`*out_state = ...`)[cite: 39].
// ============================================================================

use crate::index::QIndex;
use crate::state::QState;
use std::alloc::{alloc, dealloc, Layout};

#[repr(C)]
pub enum QStateFFI {
    Q0 = 0,
    Q1 = 1,
    Q2 = 2,
    Q3 = 3,
}

/// Kreira novi QIndex instancu. Pokazivač se mora osloboditi sa `qindex_free`.
#[unsafe(no_mangle)]
pub extern "C" fn qindex_new() -> *mut QIndex {
    Box::into_raw(Box::new(QIndex::new()))
}

/// Kreira QIndex sa zadanim kapacitetom stanja i poravnanjem.
#[unsafe(no_mangle)]
pub extern "C" fn qindex_with_capacity_and_alignment(
    capacity_states: usize,
    align: usize,
) -> *mut QIndex {
    match QIndex::try_with_capacity_and_alignment(capacity_states, align) {
        Ok(idx) => Box::into_raw(Box::new(idx)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Dodaje jedno stanje u QIndex.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qindex_push(ptr: *mut QIndex, state: u8) -> i32 {
    if ptr.is_null() || state > 3 {
        return -1; // Greška
    }
    let qindex = unsafe { &mut *ptr };
    let qstate = match state {
        0 => QState::Q0,
        1 => QState::Q1,
        2 => QState::Q2,
        3 => QState::Q3,
        _ => return -1,
    };
    qindex.push(qstate);
    0
}

/// Vraća trenutni broj stanja (len).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qindex_len(ptr: *const QIndex) -> usize {
    if ptr.is_null() { return 0; }
    unsafe { (*ptr).len() }
}

/// Vraća direktan raw pokazivač na spakovane u32 blokove (Zero-Copy za GPU/CPU).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qindex_as_raw_slice(
    ptr: *const QIndex,
    out_len: *mut usize,
) -> *const u32 {
    if ptr.is_null() {
        if !out_len.is_null() { unsafe { *out_len = 0; } }
        return std::ptr::null();
    }
    let qindex = unsafe { &*ptr };
    let slice = qindex.as_raw_slice();
    if !out_len.is_null() {
        unsafe { *out_len = slice.len(); }
    }
    slice.as_ptr()
}

/// Oslobađa QIndex memoriju dodijeljenu na Rust hipu.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qindex_free(ptr: *mut QIndex) {
    if !ptr.is_null() {
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }
}

/// Memory transfer/allocation profile intended for host RAM <-> GPU staging workflows.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMemoryProfile {
    StandardHeap = 0,
    PinnedHost = 1,
    WriteCombined = 2,
}

/// Custom allocator interface for industrial host/GPU buffer management.
pub trait QIndexAllocator {
    /// Allocates `size` bytes with `align` byte alignment for the requested memory profile.
    fn allocate(&self, size: usize, align: usize, profile: GpuMemoryProfile) -> Option<*mut u8>;

    /// Deallocates a previously allocated pointer with matching `size`, `align`, and `profile`.
    ///
    /// # Safety
    /// Caller must pass a pointer previously returned by `allocate` with the same parameters.
    unsafe fn deallocate(&self, ptr: *mut u8, size: usize, align: usize, profile: GpuMemoryProfile);
}

/// Default standard allocator wrapper implementing `QIndexAllocator`.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultQIndexAllocator;

impl QIndexAllocator for DefaultQIndexAllocator {
    fn allocate(&self, size: usize, align: usize, _profile: GpuMemoryProfile) -> Option<*mut u8> {
        if size == 0 {
            return None;
        }
        let layout = Layout::from_size_align(size, align).ok()?;
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            None
        } else {
            Some(ptr)
        }
    }

    unsafe fn deallocate(&self, ptr: *mut u8, size: usize, align: usize, _profile: GpuMemoryProfile) { unsafe {
        if ptr.is_null() || size == 0 {
            return;
        }
        if let Ok(layout) = Layout::from_size_align(size, align) {
            dealloc(ptr, layout);
        }
    }}
}

/// Staging buffer descriptor paired with a memory profile for pinned/write-combined transfers.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagedTransferBuffer {
    pub ptr: *mut u8,
    pub byte_len: usize,
    pub align: usize,
    pub profile: GpuMemoryProfile,
}

/// Allocates a raw aligned staging buffer for a given state count (2 bits per quaternary state).
#[unsafe(no_mangle)]
pub extern "C" fn qindex_alloc_staging(
    state_count: usize,
    align: usize,
    profile: GpuMemoryProfile,
) -> StagedTransferBuffer {
    let byte_len = (state_count + 3) / 4;
    let effective_align = if align == 0 { 64 } else { align };
    let allocator = DefaultQIndexAllocator;

    match allocator.allocate(byte_len, effective_align, profile) {
        Some(ptr) => StagedTransferBuffer {
            ptr,
            byte_len,
            align: effective_align,
            profile,
        },
        None => StagedTransferBuffer {
            ptr: std::ptr::null_mut(),
            byte_len: 0,
            align: effective_align,
            profile,
        },
    }
}

/// Frees a staging buffer previously allocated by `qindex_alloc_staging`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qindex_free_staging(buf: StagedTransferBuffer) {
    if buf.ptr.is_null() || buf.byte_len == 0 {
        return;
    }
    let allocator = DefaultQIndexAllocator;
    unsafe {
        allocator.deallocate(buf.ptr, buf.byte_len, buf.align, buf.profile);
    }
}

/// Status codes for consistent FFI error reporting across quaternary operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QIndexFfiStatus {
    Ok = 0,
    NullPointer = -1,
    InvalidQuatState = -2,
    OutOfBounds = -3,
    MisalignedPointer = -4,
}

unsafe impl Send for StagedTransferBuffer {}
unsafe impl Sync for StagedTransferBuffer {}

/// Validates a raw `u8` as a valid 2-bit quaternary state (`0..=3`).
#[inline]
pub const fn validate_qstate_u8(state: u8) -> Result<QState, QIndexFfiStatus> {
    match state {
        0 => Ok(QState::Q0),
        1 => Ok(QState::Q1),
        2 => Ok(QState::Q2),
        3 => Ok(QState::Q3),
        _ => Err(QIndexFfiStatus::InvalidQuatState),
    }
}

/// Checks whether a `*const QIndex` pointer is non-null and properly aligned.
#[inline]
pub fn validate_qindex_const_ptr(ptr: *const QIndex) -> Result<&'static QIndex, QIndexFfiStatus> {
    if ptr.is_null() {
        return Err(QIndexFfiStatus::NullPointer);
    }
    if (ptr as usize) % std::mem::align_of::<QIndex>() != 0 {
        return Err(QIndexFfiStatus::MisalignedPointer);
    }
    Ok(unsafe { &*ptr })
}

/// Checks whether a `*mut QIndex` pointer is non-null and properly aligned.
#[inline]
pub fn validate_qindex_mut_ptr(ptr: *mut QIndex) -> Result<&'static mut QIndex, QIndexFfiStatus> {
    if ptr.is_null() {
        return Err(QIndexFfiStatus::NullPointer);
    }
    if (ptr as usize) % std::mem::align_of::<QIndex>() != 0 {
        return Err(QIndexFfiStatus::MisalignedPointer);
    }
    Ok(unsafe { &mut *ptr })
}

/// Pushes a validated 2-bit quaternary state (`0..=3`) into `QIndex` via FFI.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qindex_push_checked(ptr: *mut QIndex, state: u8) -> QIndexFfiStatus {
    let qstate = match validate_qstate_u8(state) {
        Ok(s) => s,
        Err(err) => return err,
    };
    let qindex = match validate_qindex_mut_ptr(ptr) {
        Ok(idx) => idx,
        Err(err) => return err,
    };
    qindex.push(qstate);
    QIndexFfiStatus::Ok
}

/// Reads the quaternary state (`0..=3`) at `index` into `out_state` via FFI.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qindex_get_checked(
    ptr: *const QIndex,
    index: usize,
    out_state: *mut u8,
) -> QIndexFfiStatus {
    if out_state.is_null() {
        return QIndexFfiStatus::NullPointer;
    }
    let qindex = match validate_qindex_const_ptr(ptr) {
        Ok(idx) => idx,
        Err(err) => return err,
    };
    match qindex.get(index) {
        Some(qstate) => {
            unsafe { *out_state = qstate as u8 };
            QIndexFfiStatus::Ok
        }
        None => QIndexFfiStatus::OutOfBounds,
    }
}