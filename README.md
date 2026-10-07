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

### Benchmark Summary

| Feature / Operation | Execution Time | Key Highlight |
| :--- | :--- | :--- |
| **Allocation & Initialization** | `~1.3 µs` | Minimal overhead for cache-aligned memory setup |
| **Single Push & Retrieval** | `~1.2 µs` | Fast 2-bit state packing and bitwise extraction |
| **Batch Chunk Insertion** | `~700 ns` | Instant insertion of 16 packed states at a time |
| **Zero-Allocation Iteration** | `~1.7 µs` | Seamless traversal over packed states without heap churn |
| **Bitwise Parallel Search** | `~300 ns` | Sub-microsecond pattern matching across packed words |
| **Multi-Threading (`Send`/`Sync`)** | `~119.4 µs` | Thread-safe concurrent processing overhead |
```

### Key Performance Advantages

- **Sub-Microsecond Batching:** Pushing raw 32-bit chunks containing 16 packed states takes under **1 µs**.
- **Ultra-Fast Parallel Search:** SIMD-friendly bitwise masking searches through quantum patterns in **~300 ns**.
- **Zero-Copy Memory Access:** Provides direct slice exposure (`as_raw_slice`) ideal for direct-to-GPU DMA transfers.
- **Thread Safety:** Full `Send` and `Sync` guarantees allow concurrent access across worker threads.
```

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