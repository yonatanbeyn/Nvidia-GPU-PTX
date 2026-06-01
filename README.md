# GPU "Assembly" Hello-World — Hand-Written PTX on an RTX 4060

A minimal walkthrough of writing a kernel for an NVIDIA GPU at the lowest
levels you can practically reach: **PTX** (NVIDIA's virtual ISA, officially
supported for hand-writing) and **SASS** (the GPU's actual native machine
code, which we can read but not officially write).

## The levels of NVIDIA GPU programming

| Level | What it is | Who writes it |
|---|---|---|
| CUDA C/C++ | High-level language | Most people |
| **PTX** | NVIDIA's *virtual* ISA — text format, stable across GPU generations | You, if you want to hand-tune |
| **SASS** | Native machine code for a specific GPU architecture (e.g. `sm_89` for Ada Lovelace) | Officially: nobody (NVIDIA ships no assembler). Unofficially: tools like CuAssembler |

- `nvcc` compiles CUDA C++ → PTX → SASS.
- `ptxas` compiles PTX → SASS. We invoke it directly.
- `nvdisasm` / `cuobjdump` *read* SASS out of a cubin.

## Target hardware / toolchain

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Architecture: Ada Lovelace → `sm_89`
- Driver: 592.27
- CUDA Toolkit: 13.1 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\`)
- Host compiler: MSVC 14.29 (VS 2019 Build Tools)

## Architecture — where each piece runs

Two physical chips, two address spaces, two pieces of code we wrote. The CPU
runs `hello.exe` in user space; the kernel-mode NVIDIA driver brokers every
talk-to-the-GPU; PCIe is the wire; the GPU's SMs execute the SASS that came
out of our hand-written PTX.

```
+-------------------------------------------------------------+
|                            CPU                              |
|                                                             |
|  USER SPACE                                                 |
|  +-------------------------------------------------------+  |
|  |  hello.exe   (our host program)            <-- WE     |  |
|  |    int main() {                              WROTE    |  |
|  |       cuModuleLoad("hello.cubin");           THIS     |  |
|  |       cuLaunchKernel(...);                            |  |
|  |    }                                                  |  |
|  +-----------------------+-------------------------------+  |
|                          | function calls                   |
|                          v                                  |
|  +-------------------------------------------------------+  |
|  |  nvcuda.dll   (CUDA Driver API user-space library)    |  |
|  |    cuModuleLoad, cuMemAlloc, cuLaunchKernel, ...      |  |
|  +-----------------------+-------------------------------+  |
|                          | ioctl / WDDM DDI calls           |
|  ========================|================================  |
|  KERNEL SPACE            v                                  |
|  +-------------------------------------------------------+  |
|  |  nvlddmkm.sys  (NVIDIA kernel-mode driver)            |  |
|  |     - allocates DMA buffers in system RAM             |  |
|  |     - builds GPU command packets                      |  |
|  |     - rings a doorbell to submit work                 |  |
|  +-----------------------+-------------------------------+  |
+--------------------------|----------------------------------+
                           |
                           |  MMIO writes  +  DMA reads/writes
                           |
              =============v=========== PCIe x16 =============
                           |
                           v
+-------------------------------------------------------------+
|                  GPU — NVIDIA RTX 4060                      |
|                  Ada Lovelace, sm_89, 8 GB GDDR6            |
|                                                             |
|  +-------------------------------------------------------+  |
|  |  Front-end / GigaThread Engine                        |  |
|  |    - receives command packets from the driver         |  |
|  |    - schedules grids of thread blocks onto SMs        |  |
|  +-----------------------+-------------------------------+  |
|                          v                                  |
|  +-------------------------------------------------------+  |
|  |  Streaming Multiprocessors (SMs)                      |  |
|  |    +-----------------------------------------------+  |  |
|  |    |  warps of 32 threads execute SASS:            |  |  |
|  |    |                                               |  |  |
|  |    |    MOV   R1, c[0x0][0x28]      <-- WE WROTE   |  |  |
|  |    |    S2R   R5, SR_TID.X              THE PTX    |  |  |
|  |    |    IMAD.WIDE.U32 R2, R5, ...       THAT       |  |  |
|  |    |    STG.E [R2.64], R5               BECAME     |  |  |
|  |    |    EXIT                            THIS SASS  |  |  |
|  |    +-----------------------------------------------+  |  |
|  +-----------------------+-------------------------------+  |
|                          v                                  |
|  +-------------------------------------------------------+  |
|  |  L2 cache  +  VRAM (8 GB GDDR6)                       |  |
|  |    - holds hello.cubin's code                         |  |
|  |    - holds the output[] array we cuMemAlloc'd         |  |
|  +-------------------------------------------------------+  |
+-------------------------------------------------------------+
```

The two pieces we authored sit at opposite ends of this stack: `host.cpp` is
the topmost box on the CPU side, `hello.ptx` (compiled to SASS) is the
innermost box on the GPU side. Everything between them — nvcuda.dll, the
kernel driver, PCIe transactions, the GPU's front-end — is plumbing that
NVIDIA provides.

> On Linux the equivalent kernel-side module is `nvidia.ko` and the
> userspace library is `libcuda.so`. The shape of the stack is identical.

## Files

| File | Role |
|---|---|
| `hello.ptx` | Hand-written PTX kernel. Each thread writes its `threadIdx.x` into `out[threadIdx.x]`. |
| `host.cpp` | Minimal CUDA Driver API loader. Loads the compiled cubin, allocates GPU memory, launches the kernel with 32 threads, copies the result back, prints. |
| `hello.cubin` | Output of `ptxas` — the sm_89 binary the GPU actually runs. |
| `hello.exe` | Host executable. |

## Step-by-step

### 1. Write the PTX kernel by hand

`hello.ptx` declares a `.visible .entry` function (host-callable kernel) that
takes one 64-bit pointer parameter. The body:

```ptx
ld.param.u64       %rd1, [hello_param_out];   ; load out pointer
cvta.to.global.u64 %rd2, %rd1;                ; cast to .global address space
mov.u32            %tx,  %tid.x;              ; %tx = threadIdx.x
mul.wide.u32       %rd3, %tx, 4;              ; offset = tid * 4 bytes
add.s64            %rd4, %rd2, %rd3;          ; addr = out + offset
st.global.u32      [%rd4], %tx;               ; *addr = tid
ret;
```

Header: `.version 7.8`, `.target sm_89`, `.address_size 64`.

> Gotcha: don't name your own register `%tid` — PTX reserves `%tid` for the
> threadIdx special register, and the assembler will misparse `%tid.x`.

### 2. Assemble PTX → SASS cubin

```powershell
ptxas -arch=sm_89 -v hello.ptx -o hello.cubin
```

The `-v` flag prints register/memory usage. Expected output:

```
ptxas info : Used 8 registers, used 0 barriers, 360 bytes cmem[0]
```

### 3. Write a host loader (CUDA Driver API)

`host.cpp` uses the **Driver API** (`cuModuleLoad`, `cuLaunchKernel`) rather
than the Runtime API (`cudaLaunchKernel`) because the Driver API can load
hand-built cubins / PTX from disk. The flow:

1. `cuInit(0)` — initialize the driver.
2. `cuDeviceGet` + `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent` — attach
   to the device's primary context. (Modern replacement for `cuCtxCreate`,
   whose signature changed in CUDA 13's `_v4` revision.)
3. `cuModuleLoad("hello.cubin")` + `cuModuleGetFunction(..., "hello")`.
4. `cuMemAlloc` — allocate the output array on the GPU.
5. `cuLaunchKernel(fn, 1,1,1, 32,1,1, 0, 0, args, NULL)` — 1 block of 32 threads.
6. `cuCtxSynchronize` + `cuMemcpyDtoH` — wait, copy back, print.
7. Free / unload / release.

### 4. Build the host program

On Windows, `nvcc` needs MSVC's `cl.exe` on PATH. Easiest way is to call
`vcvars64.bat` first:

```powershell
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul && nvcc host.cpp -o hello.exe -lcuda'
```

`-lcuda` links the driver-API stub library (`cuda.lib`), not the runtime
library (`cudart`).

### 5. Run

```powershell
.\hello.exe
```

Expected output:

```
Output from PTX kernel running on sm_89:
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
```

Each value is `threadIdx.x` for that lane — all 32 threads ran in parallel and
each wrote its own index.

### 6. Disassemble cubin → SASS (what the GPU actually runs)

```powershell
nvdisasm hello.cubin
```

The kernel body (`.text.hello`) is just seven instructions:

```
/*0000*/ MOV   R1, c[0x0][0x28]            ; stack pointer setup (CUDA ABI)
/*0010*/ S2R   R5, SR_TID.X                ; R5 = threadIdx.x
/*0020*/ MOV   R2, 0x4                     ; R2 = 4
/*0030*/ ULDC.64 UR4, c[0x0][0x118]        ; uniform grid metadata
/*0040*/ IMAD.WIDE.U32 R2, R5, R2, c[0x0][0x160]
                                           ; (R2,R3) = R5*4 + out_ptr
                                           ; out_ptr is in const bank 0 at +0x160
/*0050*/ STG.E [R2.64], R5                 ; *((int*)R2_64) = R5
/*0060*/ EXIT
```

Things to notice in the diff between the PTX you wrote and the SASS that ran:

- The PTX `mul.wide.u32` + `add.s64` (two instructions) got **fused into a
  single `IMAD.WIDE.U32`** by `ptxas`. Integer multiply-add-wide is one of
  Ada's strongest pipeline units. This kind of fusion is exactly what you
  give up if you ever drop to hand-written SASS.
- The `cvta.to.global` disappeared — on Ada, kernel-parameter pointers from
  `c[0x0][...]` are already global, so no address-space conversion is needed.
- Kernel parameters live in **constant bank 0** at fixed offsets
  (`c[0x0][0x160]` for the `out` pointer). That's the CUDA ABI — useful to
  remember when reading any disassembled CUDA binary.

## Quick rebuild

```powershell
# Re-assemble PTX after editing hello.ptx
ptxas -arch=sm_89 -v hello.ptx -o hello.cubin

# Rebuild host (only needed if host.cpp changed)
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul && nvcc host.cpp -o hello.exe -lcuda'

# Run
.\hello.exe

# Inspect SASS
nvdisasm hello.cubin
```

## Further reading / next steps

- **Inline PTX in CUDA C++** via `asm volatile(...)`.
- **`nvcc -ptx -arch=sm_89 file.cu`** to see what PTX nvcc generates from
  CUDA C++ — instructive to compare against hand-written PTX.
- **CuAssembler** (https://github.com/cloudcores/CuAssembler) — community
  tool for writing real SASS by hand on Volta and newer.
- NVIDIA's official references:
  - PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/
  - CUDA Binary Utilities (SASS opcodes): https://docs.nvidia.com/cuda/cuda-binary-utilities/