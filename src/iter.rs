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