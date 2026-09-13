use std::alloc::{alloc, dealloc, Layout};
use std::fmt;
use std::ptr::NonNull;

/// 2-bitna stanja (Kvati): 00, 01, 10, 11
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QState {
    Q0 = 0b00, // 0
    Q1 = 0b01, // 1
    Q2 = 0b10, // 2
    Q3 = 0b11, // 3
}

impl From<u8> for QState {
    #[inline]
    fn from(val: u8) -> Self {
        match val & 0b11 {
            0b00 => QState::Q0,
            0b01 => QState::Q1,
            0b10 => QState::Q2,
            _ => QState::Q3,
        }
    }
}

/// Struktura indeksa optimizovana za CPU obradu i direct-to-GPU prenos.
/// Podaci su interno poravnati na 64 bajta (Cache line / DMA poravnanje).
pub struct QIndex {
    ptr: NonNull<u32>,
    capacity_u32: usize,
    len: usize, // Ukupan broj zapisanih QState elemenata (ne bajtova!)
}

impl QIndex {
    /// Konstruiše prazan QIndex sa početnim kapacitetom za elemente
    pub fn with_capacity(capacity_states: usize) -> Self {
        let u32_needed = (capacity_states + 15) / 16;
        let capacity_u32 = u32_needed.max(4); // Minimalno 4 x u32 (16 bajtova)

        // Poravnavamo memoriju na 64 bajta radi maksimalne brzine na PCIe/DMA i SIMD
        let layout = Layout::from_size_align(capacity_u32 * 4, 64)
            .expect("Failed to create memory layout");

        let raw_ptr = unsafe { alloc(layout) as *mut u32 };
        let ptr = NonNull::new(raw_ptr).expect("Memory allocation failed");

        // Inicijalizacija memorije nulama
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0, capacity_u32);
        }

        Self {
            ptr,
            capacity_u32,
            len: 0,
        }
    }

    pub fn new() -> Self {
        Self::with_capacity(64)
    }

    /// Ukupan broj skladištenih stanja (kvata)
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Ukupno zauzeće u bajtovima (sveukupno na disk/RAM/GPU)
    pub fn byte_size(&self) -> usize {
        (self.len + 3) / 4
    }

    /// Dodavanje jednog stanja u indeks: O(1)
    pub fn push(&mut self, state: QState) {
        if self.len >= self.capacity_u32 * 16 {
            self.grow();
        }

        let u32_idx = self.len / 16;       // 16 kvata staje u jedan u32 (32 bita / 2 bita)
        let bit_shift = (self.len % 16) * 2;

        unsafe {
            let elem_ptr = self.ptr.as_ptr().add(u32_idx);
            let current_val = *elem_ptr;
            *elem_ptr = current_val | ((state as u32) << bit_shift);
        }

        self.len += 1;
    }

    /// Dobijanje stanja po indeksu: O(1)
    #[inline]
    pub fn get(&self, index: usize) -> Option<QState> {
        if index >= self.len {
            return None;
        }

        let u32_idx = index / 16;
        let bit_shift = (index % 16) * 2;

        unsafe {
            let val = (*self.ptr.as_ptr().add(u32_idx) >> bit_shift) & 0b11;
            Some(QState::from(val as u8))
        }
    }

    /// Vraća sirovi pokazivač na poravnati bafer spreman za direktan prenos na GPU (DMA / PCIe)
    pub fn as_raw_slice(&self) -> &[u32] {
        let u32_used = (self.len + 15) / 16;
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), u32_used) }
    }

    /// Vraća sirovi u8 bajt isečak za serijalizaciju ili qvfs/qzip upis
    pub fn as_bytes(&self) -> &[u8] {
        let bytes_used = self.byte_size();
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const u8, bytes_used) }
    }

    /// Realokacija memorije pri popunjavanju bafera
    fn grow(&mut self) {
        let new_capacity_u32 = self.capacity_u32 * 2;
        let old_layout = Layout::from_size_align(self.capacity_u32 * 4, 64).unwrap();
        let new_layout = Layout::from_size_align(new_capacity_u32 * 4, 64).unwrap();

        unsafe {
            let new_raw_ptr = alloc(new_layout) as *mut u32;
            let new_ptr = NonNull::new(new_raw_ptr).expect("Neuspesna realokacija");
            
            // Kopiranje starih podataka
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.capacity_u32);
            // Inicijalizacija novog dela nulama
            std::ptr::write_bytes(new_ptr.as_ptr().add(self.capacity_u32), 0, self.capacity_u32);

            dealloc(self.ptr.as_ptr() as *mut u8, old_layout);
            self.ptr = new_ptr;
            self.capacity_u32 = new_capacity_u32;
        }
    }
}

impl Drop for QIndex {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity_u32 * 4, 64).unwrap();
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

impl fmt::Debug for QIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let states: Vec<QState> = (0..self.len).map(|i| self.get(i).unwrap()).collect();
        f.debug_struct("QIndex")
            .field("len", &self.len)
            .field("byte_size", &self.byte_size())
            .field("states", &states)
            .finish()
    }
}

// Omogućavamo bezbedno prebacivanje između nitima (Thread Safe)
unsafe impl Send for QIndex {}
unsafe impl Sync for QIndex {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qstate_conversion() {
        assert_eq!(QState::from(0b00), QState::Q0);
        assert_eq!(QState::from(0b01), QState::Q1);
        assert_eq!(QState::from(0b10), QState::Q2);
        assert_eq!(QState::from(0b11), QState::Q3);

        // Maska uzima samo poslednja 2 bita
        assert_eq!(QState::from(0b1111_0101), QState::Q1);
    }

    #[test]
    fn test_push_and_get_basic() {
        let mut qi = QIndex::new();
        qi.push(QState::Q0);
        qi.push(QState::Q1);
        qi.push(QState::Q2);
        qi.push(QState::Q3);

        assert_eq!(qi.len(), 4);
        assert_eq!(qi.byte_size(), 1);

        assert_eq!(qi.get(0), Some(QState::Q0));
        assert_eq!(qi.get(1), Some(QState::Q1));
        assert_eq!(qi.get(2), Some(QState::Q2));
        assert_eq!(qi.get(3), Some(QState::Q3));
        assert_eq!(qi.get(4), None); // Van granica
    }

    #[test]
    fn test_packing_in_u32_packets() {
        let mut qi = QIndex::new();

        // Upisujemo tačno 16 stanja (1 full u32 paket)
        for _ in 0..4 {
            qi.push(QState::Q0); // 00
            qi.push(QState::Q1); // 01
            qi.push(QState::Q2); // 10
            qi.push(QState::Q3); // 11
        }

        assert_eq!(qi.len(), 16);
        assert_eq!(qi.byte_size(), 4);

        let slice_u32 = qi.as_raw_slice();
        assert_eq!(slice_u32.len(), 1);

        // Proveravamo da li je bitwise reč ispravno pakovana
        // Svaki blok od 4 kvata je: 11_10_01_00 u binarnom (0xE4)
        // 0xE4E4E4E4 u hexu
        assert_eq!(slice_u32[0], 0xE4E4E4E4);
    }

    #[test]
    fn test_memory_growth_and_reallocation() {
        let mut qi = QIndex::with_capacity(16); // Malipočetni kapacitet (1 x u32)

        // Ubacujemo 100 elemenata da iznudimo višestruku realokaciju
        for i in 0..100 {
            let state = QState::from((i % 4) as u8);
            qi.push(state);
        }

        assert_eq!(qi.len(), 100);
        assert_eq!(qi.byte_size(), 25); // 100 kvata / 4 = 25 bajtova

        // Verifikujemo tačnost svih 100 upisanih vrednosti nakon premeštanja u memoriji
        for i in 0..100 {
            let expected_state = QState::from((i % 4) as u8);
            assert_eq!(qi.get(i), Some(expected_state));
        }
    }

    #[test]
    fn test_dma_64byte_alignment() {
        let qi = QIndex::new();
        let raw_ptr = qi.as_raw_slice().as_ptr() as usize;

        // Proveravamo da li je adresa alocirane memorije deljiva sa 64 (DMA poravnanje za GPU)
        assert_eq!(
            raw_ptr % 64,
            0,
            "Memory is not aligned to 64 bytes! This will result in performance degradation on PCIe."
        );
    }

    #[test]
    fn test_boundary_and_empty() {
        let qi = QIndex::new();
        assert!(qi.is_empty());
        assert_eq!(qi.len(), 0);
        assert_eq!(qi.byte_size(), 0);
        assert_eq!(qi.get(0), None);
    }

    use std::time::Instant;

#[test]
fn stress_and_performance_benchmark() {
    const TOTAL_STATES: usize = 100_000_000; // 100 Miliona kvat stanja
    println!("\n=== STARTING STRESS TEST (100.000.000 QUAT STATES) ===");

    // 1. Testiranje brzine masovnog upisa (Push Rate)
    let mut qi = QIndex::with_capacity(TOTAL_STATES);
    let start_write = Instant::now();

    for i in 0..TOTAL_STATES {
        // Rotiramo stanja 00, 01, 10, 11
        let state = QState::from((i % 4) as u8);
        qi.push(state);
    }

    let write_duration = start_write.elapsed();
    let write_throughput = (TOTAL_STATES as f64 / 1_000_000.0) / write_duration.as_secs_f64();

    println!("[1/3] ENROLLMENT completed for: {:?}", write_duration);
    println!("      Write speed: {:.2} Millions of kilowatts per second", write_throughput);

    // 2. Provera zauzeća memorije
    let bytes_used = qi.byte_size();
    let mb_used = bytes_used as f64 / (1024.0 * 1024.0);
    let raw_mb = TOTAL_STATES as f64 / (1024.0 * 1024.0);

    println!("[2/3] MEMORY EFFICIENCY:");
    println!("      Typical occupancy (1 bit/state): {:.2} MB", raw_mb);
    println!("      QIndex occupancy (2 bits/state):    {:.2} MB", mb_used);
    println!("      Savings in RAM:                   {:.1}%", (1.0 - (mb_used / raw_mb)) * 100.0);

    assert_eq!(qi.len(), TOTAL_STATES);
    assert_eq!(bytes_used, TOTAL_STATES / 4);

    // 3. Testiranje nasumičnog čitanja i provera bitwise integriteta
    let start_read = Instant::now();
    let mut errors = 0;

    // Proveravamo svaki 1000. element da izbegnemo predugo trajanje testa
    for i in (0..TOTAL_STATES).step_by(1000) {
        let expected = QState::from((i % 4) as u8);
        if qi.get(i) != Some(expected) {
            errors += 1;
        }
    }

    let read_duration = start_read.elapsed();
    println!("[3/3] Data Integrity Verification:");
    println!("      Random check time: {:?}", read_duration);
    println!("      Errors found:       {}", errors);

    assert_eq!(errors, 0, "Errors were detected in bitwise operations!");
    println!("=== The test was successfully passed! ===\n");
}

}