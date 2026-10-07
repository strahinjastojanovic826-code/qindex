/// 2-bit state (Qubit / Quantum state representation: 00, 01, 10, 11)
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    pub const unsafe fn from_u8_unchecked(val: u8) -> Self {
        // Safe because QState is #[repr(u8)] with values 0..=3
        std::mem::transmute(val & 0b11)
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