# qindex

`qindex` is a lightweight, zero-dependency Rust library designed to pack 4-state quaternary data (2 bits per state: `00`, `01`, `10`, `11`) into 64-byte aligned bitstreams for ultra-fast CPU processing and Direct Memory Access (DMA) transfers to GPUs.

By storing 4 states in a single byte, `qindex` achieves a **75% reduction in RAM and PCIe bandwidth usage**, helping eliminate the memory transfer bottleneck between Host RAM and GPU VRAM.

---

## Key Features

* **Zero Dependencies:** Built entirely with standard Rust (`std::alloc`, `std::ptr`).
* **75% RAM Savings:** Packs 4 quaternary states into 1 byte (16 states per 32-bit word).
* **64-Byte Cache-Line Alignment:** Optimized for SIMD vectorization and zero-copy DMA transfers over PCIe to GPU VRAM.
* **$O(1)$ Access Time:** Instant bitwise retrieval via fast right-shift and bitmask operations (`>>`, `&`).
* **Thread-Safe:** Implements `Send` and `Sync` for multi-threaded pipelines.

---

## Memory Efficiency

| Data Representation | Bit Width | States per Gigabyte | RAM Reduction |
| :--- | :--- | :--- | :--- |
| Standard `u32` | 32 bits | 268 Million | 0% |
| Standard `u8` | 8 bits | 1 Billion | 75% baseline |
| **`QIndex` (2-bit)** | **2 bits** | **4 Billion** | **75% saving vs `u8`** |

---

## Getting Started

Add `qindex` to your `Cargo.toml` or directly use the library in your workspace ecosystem (`qzip`, `qvfs`, `qnetwork`).

### Quick Example

```rust
use qindex::{QIndex, QState};

fn main() {
    let mut index = QIndex::new();

    // Push 2-bit states (Q0=00, Q1=01, Q2=10, Q3=11)
    index.push(QState::Q0);
    index.push(QState::Q1);
    index.push(QState::Q2);
    index.push(QState::Q3);

    assert_eq!(index.len(), 4);
    assert_eq!(index.byte_size(), 1); // 4 states packed into 1 byte

    // Access elements in O(1) time
    assert_eq!(index.get(2), Some(QState::Q2));

    // Export 64-byte aligned raw slice for direct GPU/PCIe DMA transfer
    let gpu_words: &[u32] = index.as_raw_slice();

    // Export raw bytes for qvfs storage or qzip compression
    let bytes: &[u8] = index.as_bytes();
}

GPU Integration (Compute Shader)

Because qindex packs 16 states into each 32-bit word (u32), you can send the raw buffer directly to a GPU Compute Shader (WGSL/HLSL/CUDA) and unpack the states on GPU cores with virtually zero performance overhead:

// Example WGSL compute shader function
fn unpack_qstate(packed_u32: u32, index: u32) -> u32 {
    let shift = (index % 16u) * 2u;
    return (packed_u32 >> shift) & 3u; // 3u mask (0b11)
}

Benchmarks & Testing

Run the included stress benchmark to test throughput and bitwise data integrity:

# Run tests in release mode for maximum compiler optimization
cargo test --release -- --nocapture

Performance Benchmark Results (100,000,000 States)

    RAM Usage: Reduced from 95.37 MB to 23.84 MB (75.0% memory savings).

    Data Integrity: 0 errors detected across bit boundaries.

    Throughput: ~1.4+ Billion states processed per second in release builds.