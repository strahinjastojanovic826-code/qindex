# qindex

`qindex` is a lightweight, zero-dependency Rust library designed to pack 4-state quaternary data (2 bits per state: `00`, `01`, `10`, `11`) into 64-byte aligned bitstreams for ultra-fast CPU processing and Direct Memory Access (DMA) transfers to GPUs.

By storing 4 states in a single byte, `qindex` achieves a **75% reduction in RAM and PCIe bandwidth usage**, helping eliminate the memory transfer bottleneck between Host RAM and GPU VRAM.

---

## Key Features

* **Zero Dependencies:** Built entirely with standard Rust (`std::alloc`, `std::ptr`).
* **75% RAM Savings:** Packs 4 quaternary states into 1 byte (16 states per 32-bit word).
* **Cache-Line & GPU Alignment:** 64-byte default alignment optimized for SIMD vectorization and zero-copy DMA transfers.
* **Direct GPU/VRAM Export:** Provides `as_bytes()` and `as_raw_slice()` for zero-copy PCIe buffer uploads (WebGPU, Vulkan, DirectX).
* **O(1) Access Time:** Instant bitwise retrieval and batch chunk processing (`push_u32_chunk`).
* **Safe & Fallible API:** Support for `try_with_capacity_and_alignment` alongside safe memory abstractions.
* **Thread-Safe:** Implements `Send` and `Sync` for multi-threaded pipelines.
* **Serde Integration:** Optional serialization and deserialization support for data persistence and network transfer.
* **Parallel Processing:** High-performance multi-threaded pattern matching powered by `rayon` across CPU cores.
* **Mutable Access & Iteration:** In-place state updates via `set()` and support for mutable iterators (`QIndexIterMut`).
* **RLE Compression / Sparse Representation:** Built-in Run-Length Encoding (`RleQIndex`) for ultra-efficient compression of repetitive state sequences.
* **FFI & Multi-Language Support:** C-compatible foreign function interface with ready-to-use integration examples for C, C++, and .NET (C#).
* **Quaternary Scalar & Vector Logic:** Supports per-state and packed `u32` vector operations (`q_not`, `q_and`, `q_or`, `q_xor` and `vec_not`, `vec_and`, `vec_or`, `vec_xor`) across 2-bit quaternary states.
* **Search & Filtering:** Provides `find_first`, `positions`, `filter_state`, and `filter_by` for locating and extracting states in `QIndex`.
* **Block-Segmented RLE Compression:** Extends sparse compression with `BlockRleQIndex` and `RleBlock` for fixed-size block encoding and round-trip decompression.
* **Chunked & Streaming Transfers:** Offers `QuatChunkStream` and `stream_qindex_chunks` to stream large quaternary datasets in `u32`-aligned chunks.
* **Custom Allocator & Staging Profiles:** Introduces `QIndexAllocator`, `DefaultQIndexAllocator`, `GpuMemoryProfile` (`StandardHeap`, `PinnedHost`, `WriteCombined`), and `qindex_alloc_staging` / `qindex_free_staging`.
* **Hardened FFI Validation & Thread Safety:** Adds `QIndexFfiStatus`, pointer/state validation helpers (`qindex_push_checked`, `qindex_get_checked`), and explicit `Send` / `Sync` guarantees for `GpuBufferInfo` and `StagedTransferBuffer`[cite: 16, 38].

---

## Benchmark results

| Operation / Benchmark | Workload / Scale | Measured Time | Key Metrics & Notes |
| :--- | :--- | :--- | :--- |
| **GPU Buffer Info generation** | `10,000` capacity | `200ns`[cite: 32] | Zero-copy metadata descriptor generation[cite: 12, 32] |
| **Allocation & Initialization** (`[1/6]`) | `QIndex` init | `1.7µs`[cite: 32] | Initial length: `0`, initial byte size: `3 bytes`[cite: 32] |
| **Single Push & Bitwise Retrieval** (`[2/6]`) | `8` states | `1.2µs`[cite: 32] | Pushed `8` states (`Q0..=Q3`)[cite: 19, 32] |
| **Batch Chunk Insertion** (`[3/6]`) | `16` packed `Q3` states (`0xFFFF_FFFF`) | `1.1µs`[cite: 19, 32] | New length: `24` states (`9 bytes` used for `24` 2-bit states)[cite: 32] |
| **Zero-Allocation Iteration** (`[4/6]`) | `24` items | `2.3µs`[cite: 32] | Iterated over `24` packed quaternary states[cite: 32] |
| **Bitwise Parallel Search** (`[5/6]`) | `24` states (`0xFFFF_FFFF` pattern/mask) | `500ns`[cite: 19, 32] | Found `18` `Q3` pattern matches[cite: 19, 32] |
| **Multi-Threading (`Send`/`Sync`)** (`[6/6]`) | Shared `QIndex` across thread | `175.4µs`[cite: 32] | Verified `Send` and `Sync` thread safety[cite: 14, 32] |
| **Chunk Streaming** | `100,000` states (`chunk=4096`) | `2.1µs`[cite: 32] | Streamed in `4096`-state chunks[cite: 32] |
| **FFI Checked Ops + Staging** | `50,000` pushes + staging alloc/free | `2.7693ms`[cite: 32] | Tested checked push/get, invalid/null safety, and staging buffer[cite: 32, 33] |
| **RLE Compression** | `100,000` states | `4.5761ms`[cite: 33] | Run-length encoding from `QIndex`[cite: 6, 33] |
| **RLE Decompression** | `100,000` states | `2.4668ms`[cite: 33] | Decompressed back to `QIndex` and verified equality[cite: 7, 33] |
| **Block RLE Compress + Decompress** | `100,000` states (`block_size=4096`) | `5.104ms`[cite: 33] | Segmented block RLE round-trip (`from_qindex_blocked` + `to_qindex`)[cite: 33] |
| **Vector Quaternary Logic** | `100,000` states | `277.5µs`[cite: 33] | Vector `vec_not`, `vec_and`, `vec_or`, and `vec_xor`[cite: 33] |
| **Search & Filter** | `100,000` states | `4.0715ms`[cite: 33] | `find_first`, `positions`, `filter_state`, and `filter_by`[cite: 33] |
| **Stress Test — Enrollment (`[1/2]`)** | `1,000,000` states | `17.7552ms`[cite: 22, 33] | Inserted `1,000,000` 2-bit states[cite: 22, 33] |
| **Stress Test — Data Verification (`[2/2]`)** | `1,000,000` states (`step_by(1000)`) | `60.8µs`[cite: 22, 33] | `0` errors detected in bitwise operations[cite: 22, 33] |
| **Quaternary Scalar Logic** | `500,000` iterations | `17.0403ms`[cite: 33] | Combined `q_and`, `q_or`, `q_xor`, and `q_not` operations[cite: 33] |

---

# Third-Party Licenses

This software includes open-source components developed by third parties. Below are the corresponding license notices and full legal texts for each component used in this project.

## 1. Serde

* **Author:** David Tolnay
* **Repository:** https://github.com/serde-rs/serde
* **License:** MIT / Apache 2.0 (Dual License)

### MIT License

Copyright (c) 2017 David Tolnay

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

### Apache License, Version 2.0

Copyright (c) 2017 David Tolnay

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

2. Rayon (https://github.com/rayon-rs/rayon)
   Copyright (c) 2015 Rayon contributors

You may choose to use this library under either the MIT License or the Apache 
License, Version 2.0.

OPTION A: MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

OPTION B: Apache License, Version 2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

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
    // Create an index (default 64-byte aligned for AVX-512 / CPU cache lines)
    let mut index = QIndex::new();

    // 1. Pre-allocate memory to avoid reallocations (for 100 2-bit states)
    index.reserve(100);

    // 2. Single item insertion (2-bit states: Q0=00, Q1=01, Q2=10, Q3=11)
    index.push(QState::Q0);
    index.push(QState::Q1);

    // 3. Ultra-fast batch insertion (inserts 16 packed Q3 states at once: 0xFFFF_FFFF)
    index.push_u32_chunk(0xFFFF_FFFF);

    assert_eq!(index.len(), 18);

    // 4. Fast Bitwise Pattern Matching (counts matching state patterns directly in memory)
    let pattern = 0xFFFF_FFFF; // Seeking 16 consecutive Q3 states
    let mask = 0xFFFF_FFFF;
    let matches = index.count_matches(pattern, mask);
    println!("Found {} matching states", matches);

    // 5. Idiomatic iteration over packed states
    for state in &index {
        println!("State: {:?}", state);
    }

    // 6. Export zero-copy byte slice for Direct-to-GPU DMA transfer
    let _gpu_bytes: &[u8] = index.as_bytes();
    let _gpu_u32_words: &[u32] = index.as_raw_slice();

    // 7. Reset index while retaining allocated buffer for memory reuse
    index.clear();
    assert!(index.is_empty());
}

GPU Integration (Compute Shader)

Because qindex packs 16 states into each 32-bit word (u32), you can send the raw buffer directly to a GPU Compute Shader (WGSL/HLSL/CUDA) and unpack the states on GPU cores with virtually zero performance overhead:

// Example WGSL compute shader function
fn unpack_qstate(packed_u32: u32, index: u32) -> u32 {
    let shift: u32 = (index % 16u) * 2u;
    return (packed_u32 >> shift) & 3u; // 3u = 0b11 mask
}
```

## RLE Compression and GPUBuffer

```rust
use qindex::{QIndex, QState, sparse::RleQIndex};

fn main() {
    // Create an index (default 64-byte aligned for AVX-512 / CPU cache lines)
    let mut index = QIndex::new();

    // 1. Pre-allocate memory to avoid reallocations (for 100 2-bit states)
    index.reserve(100);

    // 2. Single item insertion (2-bit states: Q0=00, Q1=01, Q2=10, Q3=11)
    index.push(QState::Q0);
    index.push(QState::Q1);

    // 3. Ultra-fast batch chunk insertion (inserts 16 packed Q3 states at once: 0xFFFF_FFFF)
    index.push_u32_chunk(0xFFFF_FFFF);

    assert_eq!(index.len(), 18);

    // 4. Fast Bitwise Pattern Matching (counts matching state patterns directly in memory)
    let pattern = 0xFFFF_FFFF; // Seeking 16 consecutive Q3 states
    let mask = 0xFFFF_FFFF;
    let matches = index.count_matches(pattern, mask);
    println!("Found {} matching states", matches);

    // 5. Idiomatic iteration over packed states
    for state in &index {
        println!("State: {:?}", state);
    }

    // 6. Export zero-copy byte slice for Direct-to-GPU DMA transfer
    let _gpu_bytes: &[u8] = index.as_bytes();
    let _gpu_u32_words: &[u32] = index.as_raw_slice();

    // 7. NEW: GPU Buffer Info (Zero-Copy metadata and strict 64-byte alignment check)
    let gpu_info = index.gpu_buffer_info();
    println!("GPU Buffer size: {} bytes, alignment: {} bytes", gpu_info.byte_size, gpu_info.alignment);

    // 8. NEW: RLE Compression (Run-Length Encoding for sparse data representation)
    let rle_compressed = RleQIndex::from_qindex(&index);
    let restored_index = rle_compressed.to_qindex();
    assert_eq!(index, restored_index);
    println!("RLE compressed into {} runs", rle_compressed.runs().len());

    // 9. Reset index while retaining allocated buffer for memory reuse
    index.clear();
    assert!(index.is_empty());
}
```

### Quaternary Logic, Search, Block RLE & Streaming

```rust
use qindex::{QIndex, QState};
use qindex::sparse::BlockRleQIndex;

fn main() {
    // 1. Scalar quaternary logic on 2-bit states (Q0..=Q3)
    let a = QState::Q2;
    let b = QState::Q1;
    let inv_a = a.q_not();
    let and_ab = a.q_and(b);
    let or_ab = a.q_or(b);
    let xor_ab = a.q_xor(b);
    println!("inv={:?}, and={:?}, or={:?}, xor={:?}", inv_a, and_ab, or_ab, xor_ab);

    // 2. Vector quaternary logic across packed indices
    let mut lhs = QIndex::with_capacity(64);
    let mut rhs = QIndex::with_capacity(64);
    for i in 0..64 {
        lhs.push(QState::from((i % 4) as u8));
        rhs.push(QState::from(((i / 2) % 4) as u8));
    }

    let _v_not = lhs.vec_not();
    let _v_and = lhs.vec_and(&rhs);
    let _v_or = lhs.vec_or(&rhs);
    let _v_xor = lhs.vec_xor(&rhs);

    // 3. Search and filtering
    let first_q2 = lhs.find_first(QState::Q2);
    let q3_positions = lhs.positions(QState::Q3);
    let only_q1 = lhs.filter_state(QState::Q1);
    let high_states = lhs.filter_by(|s| (s as u8) >= 2);
    println!(
        "first_q2={:?}, q3_count={}, only_q1_len={}, high_len={}",
        first_q2,
        q3_positions.len(),
        only_q1.len(),
        high_states.len()
    );

    // 4. Block-segmented RLE compression & decompression
    let blocked = BlockRleQIndex::from_qindex_blocked(&lhs, 16);
    let restored = blocked.to_qindex();
    assert_eq!(restored, lhs);
    println!("Block RLE blocks: {}", blocked.block_count());
}
```

### Chunked Streaming & FFI Checked Staging

```rust
use qindex::{QIndex, QState};
use qindex::ffi::{
    qindex_alloc_staging, qindex_free, qindex_free_staging, qindex_get_checked,
    qindex_new, qindex_push_checked, GpuMemoryProfile, QIndexFfiStatus,
};
use qindex::iter::stream_qindex_chunks;

fn main() {
    // 1. Streaming QIndex in fixed-size chunks
    let mut qi = QIndex::with_capacity(128);
    for i in 0..128 {
        qi.push(QState::from((i % 4) as u8));
    }

    let mut total_streamed = 0usize;
    stream_qindex_chunks(&qi, 32, |chunk| {
        total_streamed += chunk.state_count;
        println!(
            "Chunk {} at offset {} ({} states, {} u32 words)",
            chunk.chunk_index,
            chunk.state_offset,
            chunk.state_count,
            chunk.raw_u32_words.len()
        );
    });
    assert_eq!(total_streamed, qi.len());

    // 2. Checked FFI operations & staged buffer allocation
    unsafe {
        let handle = qindex_new();
        assert!(!handle.is_null());

        assert_eq!(
            qindex_push_checked(handle, 99),
            QIndexFfiStatus::InvalidQuatState
        );
        assert_eq!(qindex_push_checked(handle, 3), QIndexFfiStatus::Ok);

        let mut out_val: u8 = 0;
        assert_eq!(
            qindex_get_checked(handle, 0, &mut out_val),
            QIndexFfiStatus::Ok
        );
        assert_eq!(out_val, 3);

        let staged = qindex_alloc_staging(256, 64, GpuMemoryProfile::PinnedHost);
        if !staged.ptr.is_null() {
            qindex_free_staging(staged);
        }

        qindex_free(handle);
    }
}
```

---

### Key Performance Advantages

- **Sub-Microsecond Batching:** Pushing raw 32-bit chunks containing 16 packed states takes under **1 µs**.
- **Ultra-Fast Parallel Search:** SIMD-friendly bitwise masking searches through quantum patterns in **~300 ns**.
- **Zero-Copy Memory Access:** Provides direct slice exposure (`as_raw_slice`) ideal for direct-to-GPU DMA transfers.
- **Thread Safety:** Full `Send` and `Sync` guarantees allow concurrent access across worker threads.

---

## Usage Examples

### C

To use `qindex` in C, include `qindex.h` and link against the compiled dynamic library (`.so`, `.dll`, or `.dylib`).

```c
#include <stdio.h>
#include "qindex.h"

int main(void) {
    // Create a new QIndex instance
    QIndex* qindex = qindex_new();
    if (!qindex) {
        fprintf(stderr, "Failed to create QIndex instance\n");
        return 1;
    }

    // Push states (0 = Q0, 1 = Q1, 2 = Q2, 3 = Q3)
    qindex_push(qindex, QSTATE_Q0);
    qindex_push(qindex, QSTATE_Q1);
    qindex_push(qindex, QSTATE_Q2);
    qindex_push(qindex, QSTATE_Q3);

    printf("Total states: %zu\n", qindex_len(qindex));

    // Access raw bitmask slice (zero-copy)
    size_t slice_len = 0;
    const uint32_t* raw_slice = qindex_as_raw_slice(qindex, &slice_len);

    if (raw_slice && slice_len > 0) {
        printf("First u32 chunk: 0x%08X\n", raw_slice[0]);
    }

    // Always free the memory allocated by Rust
    qindex_free(qindex);
    return 0;
}
```

### C (Checked FFI & Staging Buffer)

```c
#include <stdio.h>
#include <stdint.h>
#include "qindex.h"

int main(void) {
    QIndex* qindex = qindex_new();
    if (!qindex) {
        fprintf(stderr, "Failed to create QIndex instance\n");
        return 1;
    }

    // Checked push validates quaternary state (0..=3)
    if (qindex_push_checked(qindex, 2) != QINDEX_FFI_STATUS_OK) {
        fprintf(stderr, "Invalid quaternary state\n");
        qindex_free(qindex);
        return 1;
    }

    // Checked read back into uint8_t
    uint8_t state_val = 0;
    if (qindex_get_checked(qindex, 0, &state_val) == QINDEX_FFI_STATUS_OK) {
        printf("State at 0: %u\n", (unsigned)state_val);
    }

    // Allocate aligned staging buffer (e.g. PinnedHost) and free it
    StagedTransferBuffer staged = qindex_alloc_staging(
        64,
        64,
        GPU_MEMORY_PROFILE_PINNED_HOST
    );
    if (staged.ptr) {
        printf("Staged buffer allocated: %zu bytes (align=%zu)\n", staged.byte_len, staged.align);
        qindex_free_staging(staged);
    }

    qindex_free(qindex);
    return 0;
}
```

---

### C++

In C++, you can wrap the raw C handle in an RAII class (`std::unique_ptr` with a custom deleter) for automatic resource management.

```cpp
#include <iostream>
#include <memory>
#include <span>
#include "qindex.h"

// RAII Deleter for C++
struct QIndexDeleter {
    void operator()(QIndex* ptr) const {
        if (ptr) qindex_free(ptr);
    }
};

using QIndexPtr = std::unique_ptr<QIndex, QIndexDeleter>;

int main() {
    // Create RAII-managed instance
    QIndexPtr qindex(qindex_new());
    if (!qindex) {
        std::cerr << "Failed to initialize QIndex" << std::endl;
        return 1;
    }

    // Push states
    qindex_push(qindex.get(), static_cast<uint8_t>(QSTATE_Q0));
    qindex_push(qindex.get(), static_cast<uint8_t>(QSTATE_Q3));

    std::cout << "State count: " << qindex_len(qindex.get()) << std::endl;

    // Get zero-copy span over packed data
    size_t slice_len = 0;
    const uint32_t* raw_ptr = qindex_as_raw_slice(qindex.get(), &slice_len);
    std::span<const uint32_t> data_span(raw_ptr, slice_len);

    for (uint32_t chunk : data_span) {
        std::cout << "Packed Chunk: " << chunk << std::endl;
    }

    // Memory is automatically released when qindex goes out of scope
    return 0;
}
```

---

### C# (.NET)

Import the `QIndex.cs` wrapper class into your project and make sure `qindex.dll` (Windows), `libqindex.so` (Linux), or `libqindex.dylib` (macOS) is in your application output directory.

```csharp
using System;
using QIndexLib;

class Program
{
    static void Main()
    {
        // 'using' block ensures SafeHandle automatically disposes Rust memory
        using (var index = new QIndex())
        {
            index.Push(QState.Q0);
            index.Push(QState.Q1);
            index.Push(QState.Q2);
            index.Push(QState.Q3);

            Console.WriteLine($"Total states: {index.Length}");

            // Access zero-copy ReadOnlySpan<uint>
            ReadOnlySpan<uint> rawData = index.GetRawSlice();
            foreach (uint chunk in rawData)
            {
                Console.WriteLine($"Packed u32 chunk: 0x{chunk:X8}");
            }
        } // Automatically releases Rust resources here
    }
}
```

## Licensing

This project is dual-licensed under **AGPLv3** and a **Commercial License**:

- **Open Source Use (AGPLv3):** Free for open-source applications and internal development under the terms of the [GNU Affero General Public License v3.0](LICENSE). Any network service or application using this library must release its source code.
- **Commercial Use:** For proprietary projects, closed-source software, commercial product embedding, or organizations that cannot comply with the AGPLv3 copyleft terms, a flexible **Commercial License** is required.

### Commercial Licensing & Support

If you need to use `qindex` without AGPLv3 obligations, or require enterprise support, custom integration, or SLA guarantees, please reach out:

- ✉️ **Contact:** `strahinjastojanovic826@gmail.com`