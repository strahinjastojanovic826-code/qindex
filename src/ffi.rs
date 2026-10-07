use crate::index::QIndex;
use crate::state::QState;
use std::slice;

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