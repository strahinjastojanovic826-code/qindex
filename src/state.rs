// ============================================================================
// UNSAFE & STATE CONVERSION SAFETY OVERVIEW (state.rs)
// ============================================================================
//
// 1. `QState::from_u8_unchecked(val: u8)`:
//    Transmutes a `u8` (`val & 0b11`) into `QState` (`#[repr(u8)]` with
//    discriminants `0..=3`)[cite: 51]. Masking with `0b11` guarantees the
//    resulting value is always in `0..=3`, making `std::mem::transmute` valid[cite: 51].
//
// 2. `QState::q_and` & `QState::q_or`:
//    Compute the minimum (`a < b`) or maximum (`a > b`) of two valid `QState`
//    values (`0..=3`) and call `Self::from_u8_unchecked(res)`[cite: 52]. Since both
//    operands are already valid `QState`s, `res` is guaranteed to be in `0..=3`[cite: 52].
//
// 3. `QState::q_xor`:
//    Performs bitwise XOR on two `QState` discriminants (`0..=3`) and passes
//    the result to `Self::from_u8_unchecked`[cite: 52]. XOR of two 2-bit values
//    (`0..=3`) always stays within `0..=3`[cite: 52].
//
// 4. `From<u8> for QState`:
//    Delegates to `Self::from_u8_unchecked(val)`, which masks `val` with `0b11`
//    before transmuting into `QState`[cite: 51, 52].
// ============================================================================

/// 2-bit state (Qubit / Quantum state representation: 00, 01, 10, 11)
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum QState {
    Q0 = 0b00,
    Q1 = 0b01,
    Q2 = 0b10,
    Q3 = 0b11,
}

impl QState {
    /// Creates a QState from raw 2-bit value without checking boundary.
    ///
    /// # Safety
    /// `val & 0b11` must be guaranteed by the caller or masked properly.
    #[inline]
    pub const unsafe fn from_u8_unchecked(val: u8) -> Self { unsafe {
        // Safe because QState is #[repr(u8)] with values 0..=3
        std::mem::transmute(val & 0b11)
    }}

    /// Quaternary logical NOT (inversion over 2-bit state: 0b00 <-> 0b11, 0b01 <-> 0b10).
    #[inline]
    pub const fn q_not(self) -> Self {
        match self as u8 {
            0b00 => Self::Q3,
            0b01 => Self::Q2,
            0b10 => Self::Q1,
            _ => Self::Q0,
        }
    }

    /// Quaternary logical AND (min on {0, 1, 2, 3}).
    #[inline]
    pub const fn q_and(self, other: Self) -> Self {
        let a = self as u8;
        let b = other as u8;
        let res = if a < b { a } else { b };
        unsafe { Self::from_u8_unchecked(res) }
    }

    /// Quaternary logical OR (max on {0, 1, 2, 3}).
    #[inline]
    pub const fn q_or(self, other: Self) -> Self {
        let a = self as u8;
        let b = other as u8;
        let res = if a > b { a } else { b };
        unsafe { Self::from_u8_unchecked(res) }
    }

    /// Quaternary bitwise XOR on 2-bit states.
    #[inline]
    pub const fn q_xor(self, other: Self) -> Self {
        unsafe { Self::from_u8_unchecked((self as u8) ^ (other as u8)) }
    }
    
}

impl From<u8> for QState {
    #[inline]
    fn from(val: u8) -> Self {
        unsafe { Self::from_u8_unchecked(val) }
    }
}

impl From<QState> for u8 {
    #[inline]
    fn from(state: QState) -> Self {
        state as u8
    }

}