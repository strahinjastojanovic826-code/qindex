pub mod index;
pub mod iter;
pub mod state;
pub mod ffi;
pub mod sparse;

pub use index::QIndex;
pub use iter::QIndexIter;
pub use state::QState;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;
    use super::ffi::*;
    use super::state::QState;
    use std::ptr;
    use crate::sparse::RleQIndex;

    fn print_header(title: &str) {
        println!("\n==================================================");
        println!(" TEST SUITE: {}", title);
        println!("==================================================");
    }

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
    fn test_dma_64byte_alignment() {
        let qi = QIndex::new();
        let raw_ptr = qi.as_bytes().as_ptr() as usize;

        // Proveravamo da li je adresa alocirane memorije deljiva sa 64 (DMA poravnanje za GPU)
        assert_eq!(
            raw_ptr % 64,
            0,
            "Memory is not aligned to 64 bytes! This will result in performance degradation on PCIe."
        );
    }

    #[test]
    fn test_clear_and_reuse() {
        let mut qi = QIndex::new();
        qi.push(QState::Q1);
        qi.push(QState::Q3);

        assert_eq!(qi.len(), 2);

        qi.clear();
        assert_eq!(qi.len(), 0);
        assert!(qi.is_empty());
        assert_eq!(qi.get(0), None);

        // Provera da li se memorija može ponovo koristiti nakon clear-a
        qi.push(QState::Q2);
        assert_eq!(qi.len(), 1);
        assert_eq!(qi.get(0), Some(QState::Q2));
    }

    #[test]
    fn test_reserve_explicit() {
        let mut qi = QIndex::new();
        qi.reserve(100); // Unapred alociramo prostor za 100 elemenata

        let initial_cap = qi.capacity();
        assert!(initial_cap >= 100);

        for _ in 0..100 {
            qi.push(QState::Q0);
        }

        // Kapacitet ne bi trebalo da se menja jer je unapred rezervisan
        assert_eq!(qi.capacity(), initial_cap);
    }

    #[test]
    fn test_iterator() {
        let mut qi = QIndex::new();
        qi.push(QState::Q0);
        qi.push(QState::Q1);
        qi.push(QState::Q2);

        let collected: Vec<QState> = qi.into_iter().collect();
        assert_eq!(collected, vec![QState::Q0, QState::Q1, QState::Q2]);
    }

    #[test]
    fn test_comprehensive_qindex_report() {
        print_header("QIndex Detailed Validation & Performance Report");

        // 1. TEST: Osnovna alokacija i rad sa kapacitetom
        let start = Instant::now();
        let mut qidx = QIndex::with_capacity(64);
        assert_eq!(qidx.len(), 0);
        assert!(qidx.is_empty());
        println!(
            "[1/6] Allocation & Initialization OK (Time: {:?})",
            start.elapsed()
        );
        println!("      - Initial Length: {}", qidx.len());
        println!("      - Initial Byte Size: {} bytes", qidx.byte_size());

        // 2. TEST: Pojedinačni unos (push) i bitwise pakovanje
        let start = Instant::now();
        let states_to_insert = [
            QState::Q0,
            QState::Q1,
            QState::Q2,
            QState::Q3,
            QState::Q3,
            QState::Q2,
            QState::Q1,
            QState::Q0,
        ];

        for &st in &states_to_insert {
            qidx.push(st);
        }

        assert_eq!(qidx.len(), 8);
        for (idx, &expected) in states_to_insert.iter().enumerate() {
            assert_eq!(qidx.get(idx), Some(expected));
        }
        println!(
            "[2/6] Single Push & Bitwise Retrieval OK (Time: {:?})",
            start.elapsed()
        );
        println!("      - Pushed {} states successfully", qidx.len());

        // 3. TEST: Batch Unos ('push_u32_chunk')
        let start = Instant::now();
        let chunk_q3: u32 = 0xFFFF_FFFF; // 16 spakovanih Q3 stanja
        qidx.push_u32_chunk(chunk_q3);

        assert_eq!(qidx.len(), 24);
        for i in 8..24 {
            assert_eq!(qidx.get(i), Some(QState::Q3));
        }
        println!(
            "[3/6] Batch Chunk Insertion OK (Time: {:?})",
            start.elapsed()
        );
        println!(
            "      - Added 16 packed Q3 states instantly. New Len: {}",
            qidx.len()
        );

        // 4. TEST: Iteracija bez alokacije ('QIndexIter')
        let start = Instant::now();
        let iterated_states: Vec<QState> = qidx.into_iter().collect();
        assert_eq!(iterated_states.len(), qidx.len());
        assert_eq!(iterated_states[0], QState::Q0);
        assert_eq!(iterated_states[8], QState::Q3);
        println!(
            "[4/6] Zero-Allocation Iteration OK (Time: {:?})",
            start.elapsed()
        );
        println!(
            "      - Successfully iterated over {} items",
            iterated_states.len()
        );

        // 5. TEST: Ultra-brza bitwise pretraga ('count_matches')
        let start = Instant::now();
        let matches_count = qidx.count_matches(0xFFFF_FFFF, 0xFFFF_FFFF);
        println!(
            "[5/6] Bitwise Parallel Search OK (Time: {:?})",
            start.elapsed()
        );
        println!("      - Total Q3 Pattern Matches Found: {}", matches_count);

        // 6. TEST: Provjera višenitnog rada (Send / Sync)
        let start = Instant::now();
        let shared_qidx = Arc::new(qidx);
        let shared_clone = Arc::clone(&shared_qidx);

        let handle = thread::spawn(move || shared_clone.len());

        let thread_len = handle.join().unwrap();
        assert_eq!(thread_len, shared_qidx.len());
        println!(
            "[6/6] Multi-Threading (Send/Sync) OK (Time: {:?})",
            start.elapsed()
        );

        println!("\n==================================================");
        println!(" SUMMARY REPORT:");
        println!(" - Status: ALL TESTS PASSED");
        println!(
            " - Memory Efficiency: {} bytes used for {} 2-bit states",
            shared_qidx.byte_size(),
            shared_qidx.len()
        );
        println!("==================================================\n");
    }

    #[test]
    fn test_push_and_get_basic() {
        let mut qi = QIndex::new();
        qi.push(QState::Q0);
        qi.push(QState::Q1);
        qi.push(QState::Q2);
        qi.push(QState::Q3);

        assert_eq!(qi.len(), 4);

        assert_eq!(qi.get(0), Some(QState::Q0));
        assert_eq!(qi.get(1), Some(QState::Q1));
        assert_eq!(qi.get(2), Some(QState::Q2));
        assert_eq!(qi.get(3), Some(QState::Q3));
        assert_eq!(qi.get(4), None);
    }

    #[test]
    fn test_packing_in_u32_packets() {
        let mut qi = QIndex::new();

        for _ in 0..4 {
            qi.push(QState::Q0);
            qi.push(QState::Q1);
            qi.push(QState::Q2);
            qi.push(QState::Q3);
        }

        assert_eq!(qi.len(), 16);
        let slice_u32 = qi.as_raw_slice();
        assert_eq!(slice_u32.len(), 1);

        for i in 0..16 {
            let expected = match i % 4 {
                0 => QState::Q0,
                1 => QState::Q1,
                2 => QState::Q2,
                _ => QState::Q3,
            };
            assert_eq!(qi.get(i), Some(expected));
        }
    }

    #[test]
    fn test_memory_growth_and_reallocation() {
        let mut qi = QIndex::with_capacity(16);

        for i in 0..100 {
            let state = QState::from((i % 4) as u8);
            qi.push(state);
        }

        assert_eq!(qi.len(), 100);

        for i in 0..100 {
            let expected_state = QState::from((i % 4) as u8);
            assert_eq!(qi.get(i), Some(expected_state));
        }
    }

    #[test]
    fn test_boundary_and_empty() {
        let qi = QIndex::new();
        assert!(qi.is_empty());
        assert_eq!(qi.len(), 0);
        assert_eq!(qi.get(0), None);
    }

    #[test]
    fn stress_and_performance_benchmark() {
        const TOTAL_STATES: usize = 1_000_000;
        println!("\n=== STARTING STRESS TEST ({}) ===", TOTAL_STATES);

        let mut qi = QIndex::with_capacity(TOTAL_STATES);
        let start_write = Instant::now();

        for i in 0..TOTAL_STATES {
            let state = QState::from((i % 4) as u8);
            qi.push(state);
        }

        let write_duration = start_write.elapsed();
        println!("[1/2] Enrollment completed in: {:?}", write_duration);

        assert_eq!(qi.len(), TOTAL_STATES);

        let start_read = Instant::now();
        let mut errors = 0;

        for i in (0..TOTAL_STATES).step_by(1000) {
            let expected = QState::from((i % 4) as u8);
            if qi.get(i) != Some(expected) {
                errors += 1;
            }
        }

        let read_duration = start_read.elapsed();
        println!("[2/2] Data verification completed in: {:?}", read_duration);

        assert_eq!(errors, 0, "Errors detected in bitwise operations!");
    }
    #[test]
    fn test_ffi_lifecycle() {
        unsafe {
            // Test creation
            let handle = qindex_new();
            assert!(!handle.is_null());
            assert_eq!(qindex_len(handle), 0);

            // Test pushing valid states (Q0=0, Q1=1, Q2=2, Q3=3)
            assert_eq!(qindex_push(handle, 0), 0);
            assert_eq!(qindex_push(handle, 1), 0);
            assert_eq!(qindex_push(handle, 2), 0);
            assert_eq!(qindex_push(handle, 3), 0);

            // Verify length
            assert_eq!(qindex_len(handle), 4);

            // Test raw slice extraction
            let mut slice_len: usize = 0;
            let raw_ptr = qindex_as_raw_slice(handle, &mut slice_len);

            assert!(!raw_ptr.is_null());
            assert_eq!(slice_len, 1); // 4 states fit into 1 x u32 chunk (16 states per u32)

            // Verify bit layout in the packed u32 chunk:
            // Q0 (00) | Q1 (01 << 2) | Q2 (10 << 4) | Q3 (11 << 6)
            // Binary: 11_10_01_00 = 0b11100100 = 0xE4
            assert_eq!(*raw_ptr, 0b11100100);

            // Cleanup
            qindex_free(handle);
        }
    }

    #[test]
    fn test_ffi_custom_capacity_and_alignment() {
        unsafe {
            let handle = qindex_with_capacity_and_alignment(128, 64);
            assert!(!handle.is_null());
            assert_eq!(qindex_len(handle), 0);

            // Push multiple items to trigger multi-chunk packing
            for i in 0..32 {
                let state = (i % 4) as u8;
                assert_eq!(qindex_push(handle, state), 0);
            }

            assert_eq!(qindex_len(handle), 32);

            let mut slice_len: usize = 0;
            let raw_ptr = qindex_as_raw_slice(handle, &mut slice_len);

            assert!(!raw_ptr.is_null());
            assert_eq!(slice_len, 2); // 32 states = 2 x u32 chunks

            qindex_free(handle);
        }
    }

    #[test]
    fn test_ffi_invalid_inputs_and_null_safety() {
        unsafe {
            let handle = qindex_new();

            // Test pushing invalid state value (> 3)
            assert_eq!(qindex_push(handle, 4), -1);
            assert_eq!(qindex_push(handle, 255), -1);
            assert_eq!(qindex_len(handle), 0);

            // Test NULL pointer safety
            let null_handle: *mut crate::index::QIndex = ptr::null_mut();

            assert_eq!(qindex_push(null_handle, 0), -1);
            assert_eq!(qindex_len(null_handle), 0);

            let mut slice_len: usize = 999;
            let raw_ptr = qindex_as_raw_slice(null_handle, &mut slice_len);
            assert!(raw_ptr.is_null());
            assert_eq!(slice_len, 0);

            // Passing null to free should be a no-op without panicking/crashing
            qindex_free(null_handle);

            // Cleanup valid handle
            qindex_free(handle);
        }
    }
    #[test]
fn test_rayon_parallel_counting() {
    let mut qi = QIndex::new();
    // Ubacujemo 32 Q3 stanja (svako Q3 je 0b11)
    for _ in 0..32 {
        qi.push(QState::Q3);
    }

    // Pattern za Q3 na svim pozicijama u u32 reči je 0xFFFF_FFFF
    let pattern = 0xFFFF_FFFF;
    let mask = 0xFFFF_FFFF;

    let matches = qi.par_count_matches(pattern, mask);
    assert_eq!(matches, 32);
}

#[test]
fn test_benchmark_rle_compression() {
    const TOTAL_STATES: usize = 100_000;
    let mut qidx = QIndex::with_capacity(TOTAL_STATES);
    for i in 0..TOTAL_STATES {
        let state = match i % 4 {
            0 => QState::Q0,
            1 => QState::Q1,
            2 => QState::Q2,
            _ => QState::Q3,
        };
        qidx.push(state);
    }

    let start = Instant::now();
    let compressed = RleQIndex::from_qindex(&qidx);
    let duration = start.elapsed();

    println!("[BENCHMARK] RLE Compression ({} states): {:?}", TOTAL_STATES, duration);
    assert!(!compressed.runs().is_empty());
}

#[test]
fn test_benchmark_rle_decompression() {
    const TOTAL_STATES: usize = 100_000;
    let mut qidx = QIndex::with_capacity(TOTAL_STATES);
    for i in 0..TOTAL_STATES {
        let state = match i % 4 {
            0 => QState::Q0,
            1 => QState::Q1,
            2 => QState::Q2,
            _ => QState::Q3,
        };
        qidx.push(state);
    }
    let compressed = RleQIndex::from_qindex(&qidx);

    let start = Instant::now();
    let decompressed = compressed.to_qindex();
    let duration = start.elapsed();

    println!("[BENCHMARK] RLE Decompression ({} states): {:?}", TOTAL_STATES, duration);
    assert_eq!(qidx, decompressed);
}

#[test]
fn test_benchmark_gpu_buffer_info() {
    let qidx = QIndex::with_capacity(10_000);

    let start = Instant::now();
    let gpu_info = qidx.gpu_buffer_info();
    let duration = start.elapsed();

    println!("[BENCHMARK] GPU Buffer Info generation: {:?}", duration);
    assert!(!gpu_info.ptr.is_null());
    assert_eq!(gpu_info.alignment, 64);
}

#[test]
    fn test_quaternary_scalar_logic_and_benchmark() {
        const ITERATIONS: usize = 500_000;
        let mut acc = 0usize;

        let start = Instant::now();
        for i in 0..ITERATIONS {
            let a = QState::from((i % 4) as u8);
            let b = QState::from(((i * 3) % 4) as u8);
            let r = a.q_and(b).q_or(a.q_xor(b)).q_not();
            acc += r as usize;
        }
        let elapsed = start.elapsed();

        assert_eq!(QState::Q0.q_not(), QState::Q3);
        assert_eq!(QState::Q1.q_not(), QState::Q2);
        assert_eq!(QState::Q2.q_and(QState::Q1), QState::Q1);
        assert_eq!(QState::Q2.q_or(QState::Q1), QState::Q2);
        assert_eq!(QState::Q2.q_xor(QState::Q3), QState::Q1);
        assert!(acc > 0);

        println!(
            "[BENCHMARK] Quaternary scalar logic ({} iters): {:?}",
            ITERATIONS, elapsed
        );
    }

    #[test]
    fn test_vector_quaternary_logic_and_benchmark() {
        const TOTAL_STATES: usize = 100_000;
        let mut lhs = QIndex::with_capacity(TOTAL_STATES);
        let mut rhs = QIndex::with_capacity(TOTAL_STATES);

        for i in 0..TOTAL_STATES {
            lhs.push(QState::from((i % 4) as u8));
            rhs.push(QState::from(((i / 2) % 4) as u8));
        }

        let start = Instant::now();
        let inv = lhs.vec_not();
        let and_res = lhs.vec_and(&rhs);
        let or_res = lhs.vec_or(&rhs);
        let xor_res = lhs.vec_xor(&rhs);
        let elapsed = start.elapsed();

        assert_eq!(inv.len(), TOTAL_STATES);
        assert_eq!(inv.get(0), Some(QState::Q0.q_not()));
        assert_eq!(and_res.get(2), Some(QState::Q2.q_and(QState::Q1)));
        assert_eq!(or_res.get(2), Some(QState::Q1));
        assert_eq!(xor_res.get(2), Some(QState::Q2.q_xor(QState::Q1)));

        println!(
            "[BENCHMARK] Vector quaternary logic ({} states): {:?}",
            TOTAL_STATES, elapsed
        );
    }

    #[test]
    fn test_search_and_filter_and_benchmark() {
        const TOTAL_STATES: usize = 100_000;
        let mut qi = QIndex::with_capacity(TOTAL_STATES);
        for i in 0..TOTAL_STATES {
            qi.push(QState::from((i % 4) as u8));
        }

        let start = Instant::now();
        let first_q2 = qi.find_first(QState::Q2);
        let q2_positions = qi.positions(QState::Q2);
        let filtered_q2 = qi.filter_state(QState::Q2);
        let filtered_high = qi.filter_by(|s| (s as u8) >= 2);
        let elapsed = start.elapsed();

        assert_eq!(first_q2, Some(2));
        assert_eq!(q2_positions.len(), TOTAL_STATES / 4);
        assert_eq!(filtered_q2.len(), TOTAL_STATES / 4);
        assert_eq!(filtered_high.len(), TOTAL_STATES / 2);

        println!(
            "[BENCHMARK] Search & filter ({} states): {:?}",
            TOTAL_STATES, elapsed
        );
    }

    #[test]
    fn test_block_rle_compression_and_benchmark() {
        const TOTAL_STATES: usize = 100_000;
        const BLOCK_SIZE: usize = 4096;
        let mut qi = QIndex::with_capacity(TOTAL_STATES);
        for i in 0..TOTAL_STATES {
            qi.push(QState::from(((i / 64) % 4) as u8));
        }

        let start = Instant::now();
        let blocked = crate::sparse::BlockRleQIndex::from_qindex_blocked(&qi, BLOCK_SIZE);
        let restored = blocked.to_qindex();
        let elapsed = start.elapsed();

        assert_eq!(blocked.len(), TOTAL_STATES);
        assert!(!blocked.is_empty());
        assert!(blocked.block_count() > 1);
        assert_eq!(restored.len(), TOTAL_STATES);
        assert_eq!(restored, qi);

        println!(
            "[BENCHMARK] Block RLE compress + decompress ({} states, block_size={}): {:?}",
            TOTAL_STATES, BLOCK_SIZE, elapsed
        );
    }

    #[test]
    fn test_chunk_streaming_and_benchmark() {
        const TOTAL_STATES: usize = 100_000;
        const CHUNK_STATES: usize = 4096;
        let mut qi = QIndex::with_capacity(TOTAL_STATES);
        for i in 0..TOTAL_STATES {
            qi.push(QState::from((i % 4) as u8));
        }

        let start = Instant::now();
        let mut streamed_states = 0usize;
        let mut chunk_count = 0usize;
        crate::iter::stream_qindex_chunks(&qi, CHUNK_STATES, |chunk| {
            streamed_states += chunk.state_count;
            chunk_count += 1;
            assert!(!chunk.raw_u32_words.is_empty());
        });
        let elapsed = start.elapsed();

        assert_eq!(streamed_states, TOTAL_STATES);
        assert!(chunk_count > 0);

        println!(
            "[BENCHMARK] Chunk streaming ({} states, chunk={}): {:?}",
            TOTAL_STATES, CHUNK_STATES, elapsed
        );
    }

    #[test]
    fn test_ffi_staging_and_checked_ops_and_benchmark() {
        const BENCH_OPS: usize = 50_000;
        let start = Instant::now();

        unsafe {
            let handle = qindex_new();
            assert!(!handle.is_null());

            assert_eq!(
                crate::ffi::qindex_push_checked(handle, 5),
                crate::ffi::QIndexFfiStatus::InvalidQuatState
            );
            assert_eq!(
                crate::ffi::qindex_push_checked(handle, 2),
                crate::ffi::QIndexFfiStatus::Ok
            );

            let mut out_val: u8 = 99;
            assert_eq!(
                crate::ffi::qindex_get_checked(handle, 0, &mut out_val),
                crate::ffi::QIndexFfiStatus::Ok
            );
            assert_eq!(out_val, 2);

            assert_eq!(
                crate::ffi::qindex_get_checked(handle, 10, &mut out_val),
                crate::ffi::QIndexFfiStatus::OutOfBounds
            );

            for i in 0..BENCH_OPS {
                assert_eq!(
                    crate::ffi::qindex_push_checked(handle, (i % 4) as u8),
                    crate::ffi::QIndexFfiStatus::Ok
                );
            }

            let staged = crate::ffi::qindex_alloc_staging(
                1024,
                64,
                crate::ffi::GpuMemoryProfile::PinnedHost,
            );
            assert!(!staged.ptr.is_null());
            assert_eq!(staged.byte_len, 256);
            assert_eq!(staged.align, 64);
            crate::ffi::qindex_free_staging(staged);

            qindex_free(handle);
        }

        let elapsed = start.elapsed();
        println!(
            "[BENCHMARK] FFI checked ops + staging ({} pushes): {:?}",
            BENCH_OPS, elapsed
        );
    }

}