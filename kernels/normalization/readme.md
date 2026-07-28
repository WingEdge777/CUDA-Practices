# norm 1d

## Overview

norm kernels calculate y = (x - mean) / std

- [x] naive Torch norm
- [x] norm — FP32
- [x] norm — FP32x4
- [x] norm — FP16
- [x] norm — FP16x8 packed
- [x] norm — FP32x4 split-k
- [x] pytorch op bindings && diff check

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### Sample output

#### test split k

```bash
####################################################################################################
n: 1, m: 8192
torch                          mean time: 0.007360 ms
norm_fp32                      mean time: 0.009280 ms, speedup: 0.79
norm_fp32x4                    mean time: 0.008352 ms, speedup: 0.88
norm_fp32x4_split_k            mean time: 0.011408 ms, speedup: 0.65
####################################################################################################
n: 1, m: 12800
torch                          mean time: 0.021808 ms
norm_fp32                      mean time: 0.010592 ms, speedup: 2.06
norm_fp32x4                    mean time: 0.009136 ms, speedup: 2.39
norm_fp32x4_split_k            mean time: 0.011392 ms, speedup: 1.91
####################################################################################################
n: 1, m: 102400
torch                          mean time: 0.038160 ms
norm_fp32                      mean time: 0.077584 ms, speedup: 0.49
norm_fp32x4                    mean time: 0.031840 ms, speedup: 1.20
norm_fp32x4_split_k            mean time: 0.012736 ms, speedup: 3.00
####################################################################################################
n: 4, m: 8192
torch                          mean time: 0.011184 ms
norm_fp32                      mean time: 0.009280 ms, speedup: 1.21
norm_fp32x4                    mean time: 0.008896 ms, speedup: 1.26
norm_fp32x4_split_k            mean time: 0.011392 ms, speedup: 0.98
####################################################################################################
n: 4, m: 12800
torch                          mean time: 0.013664 ms
norm_fp32                      mean time: 0.011312 ms, speedup: 1.21
norm_fp32x4                    mean time: 0.009392 ms, speedup: 1.45
norm_fp32x4_split_k            mean time: 0.011632 ms, speedup: 1.17
####################################################################################################
n: 4, m: 102400
torch                          mean time: 0.072192 ms
norm_fp32                      mean time: 0.081824 ms, speedup: 0.88
norm_fp32x4                    mean time: 0.036272 ms, speedup: 1.99
norm_fp32x4_split_k            mean time: 0.018128 ms, speedup: 3.98
####################################################################################################
n: 8, m: 8192
torch                          mean time: 0.011248 ms
norm_fp32                      mean time: 0.009296 ms, speedup: 1.21
norm_fp32x4                    mean time: 0.009216 ms, speedup: 1.22
norm_fp32x4_split_k            mean time: 0.012080 ms, speedup: 0.93
####################################################################################################
n: 8, m: 12800
torch                          mean time: 0.013600 ms
norm_fp32                      mean time: 0.011888 ms, speedup: 1.14
norm_fp32x4                    mean time: 0.010832 ms, speedup: 1.26
norm_fp32x4_split_k            mean time: 0.012864 ms, speedup: 1.06
####################################################################################################
n: 8, m: 102400
torch                          mean time: 0.072480 ms
norm_fp32                      mean time: 0.088896 ms, speedup: 0.82
norm_fp32x4                    mean time: 0.043936 ms, speedup: 1.65
norm_fp32x4_split_k            mean time: 0.026400 ms, speedup: 2.75
####################################################################################################
n: 16, m: 8192
torch                          mean time: 0.011344 ms
norm_fp32                      mean time: 0.011376 ms, speedup: 1.00
norm_fp32x4                    mean time: 0.009568 ms, speedup: 1.19
norm_fp32x4_split_k            mean time: 0.013504 ms, speedup: 0.84
####################################################################################################
n: 16, m: 12800
torch                          mean time: 0.014032 ms
norm_fp32                      mean time: 0.013488 ms, speedup: 1.04
norm_fp32x4                    mean time: 0.013312 ms, speedup: 1.05
norm_fp32x4_split_k            mean time: 0.017440 ms, speedup: 0.80
####################################################################################################
n: 16, m: 102400
torch                          mean time: 0.079440 ms
norm_fp32                      mean time: 0.100480 ms, speedup: 0.79
norm_fp32x4                    mean time: 0.050576 ms, speedup: 1.57
norm_fp32x4_split_k            mean time: 0.041776 ms, speedup: 1.90
```

#### test all

```bash
####################################################################################################
n: 64, m: 2048
torch                          mean time: 0.009440 ms
norm_fp32                      mean time: 0.009824 ms, speedup: 0.96
norm_fp32x4                    mean time: 0.011360 ms, speedup: 0.83
norm_fp16                      mean time: 0.011216 ms, speedup: 0.84
norm_fp16x8_packed             mean time: 0.010704 ms, speedup: 0.88
####################################################################################################
n: 64, m: 4096
torch                          mean time: 0.011040 ms
norm_fp32                      mean time: 0.013360 ms, speedup: 0.83
norm_fp32x4                    mean time: 0.013456 ms, speedup: 0.82
norm_fp16                      mean time: 0.013472 ms, speedup: 0.82
norm_fp16x8_packed             mean time: 0.011488 ms, speedup: 0.96
####################################################################################################
n: 64, m: 8192
torch                          mean time: 0.013696 ms
norm_fp32                      mean time: 0.021792 ms, speedup: 0.63
norm_fp32x4                    mean time: 0.019312 ms, speedup: 0.71
norm_fp16                      mean time: 0.019776 ms, speedup: 0.69
norm_fp16x8_packed             mean time: 0.013360 ms, speedup: 1.03
####################################################################################################
n: 64, m: 12800
torch                          mean time: 0.017776 ms
norm_fp32                      mean time: 0.031904 ms, speedup: 0.56
norm_fp32x4                    mean time: 0.021376 ms, speedup: 0.83
norm_fp16                      mean time: 0.025568 ms, speedup: 0.70
norm_fp16x8_packed             mean time: 0.014768 ms, speedup: 1.20
####################################################################################################
n: 128, m: 2048
torch                          mean time: 0.009536 ms
norm_fp32                      mean time: 0.008160 ms, speedup: 1.17
norm_fp32x4                    mean time: 0.011120 ms, speedup: 0.86
norm_fp16                      mean time: 0.009792 ms, speedup: 0.97
norm_fp16x8_packed             mean time: 0.008992 ms, speedup: 1.06
####################################################################################################
n: 128, m: 4096
torch                          mean time: 0.012976 ms
norm_fp32                      mean time: 0.014944 ms, speedup: 0.87
norm_fp32x4                    mean time: 0.014768 ms, speedup: 0.88
norm_fp16                      mean time: 0.012272 ms, speedup: 1.06
norm_fp16x8_packed             mean time: 0.011168 ms, speedup: 1.16
####################################################################################################
n: 128, m: 8192
torch                          mean time: 0.017920 ms
norm_fp32                      mean time: 0.029600 ms, speedup: 0.61
norm_fp32x4                    mean time: 0.026992 ms, speedup: 0.66
norm_fp16                      mean time: 0.019584 ms, speedup: 0.92
norm_fp16x8_packed             mean time: 0.014976 ms, speedup: 1.20
####################################################################################################
n: 128, m: 12800
torch                          mean time: 0.041424 ms
norm_fp32                      mean time: 0.049440 ms, speedup: 0.84
norm_fp32x4                    mean time: 0.049520 ms, speedup: 0.84
norm_fp16                      mean time: 0.034992 ms, speedup: 1.18
norm_fp16x8_packed             mean time: 0.021488 ms, speedup: 1.93
####################################################################################################
n: 512, m: 2048
torch                          mean time: 0.019104 ms
norm_fp32                      mean time: 0.025600 ms, speedup: 0.75
norm_fp32x4                    mean time: 0.025552 ms, speedup: 0.75
norm_fp16                      mean time: 0.019120 ms, speedup: 1.00
norm_fp16x8_packed             mean time: 0.014560 ms, speedup: 1.31
####################################################################################################
n: 512, m: 4096
torch                          mean time: 0.048064 ms
norm_fp32                      mean time: 0.056416 ms, speedup: 0.85
norm_fp32x4                    mean time: 0.054464 ms, speedup: 0.88
norm_fp16                      mean time: 0.036608 ms, speedup: 1.31
norm_fp16x8_packed             mean time: 0.030144 ms, speedup: 1.59
####################################################################################################
n: 512, m: 8192
torch                          mean time: 0.103648 ms
norm_fp32                      mean time: 0.106768 ms, speedup: 0.97
norm_fp32x4                    mean time: 0.103792 ms, speedup: 1.00
norm_fp16                      mean time: 0.062496 ms, speedup: 1.66
norm_fp16x8_packed             mean time: 0.053552 ms, speedup: 1.94
####################################################################################################
n: 512, m: 12800
torch                          mean time: 0.185216 ms
norm_fp32                      mean time: 0.169520 ms, speedup: 1.09
norm_fp32x4                    mean time: 0.165104 ms, speedup: 1.12
norm_fp16                      mean time: 0.094672 ms, speedup: 1.96
norm_fp16x8_packed             mean time: 0.084624 ms, speedup: 2.19
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.047648 ms
norm_fp32                      mean time: 0.054416 ms, speedup: 0.88
norm_fp32x4                    mean time: 0.052608 ms, speedup: 0.91
norm_fp16                      mean time: 0.033248 ms, speedup: 1.43
norm_fp16x8_packed             mean time: 0.025280 ms, speedup: 1.88
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.111424 ms
norm_fp32                      mean time: 0.111280 ms, speedup: 1.00
norm_fp32x4                    mean time: 0.110336 ms, speedup: 1.01
norm_fp16                      mean time: 0.060320 ms, speedup: 1.85
norm_fp16x8_packed             mean time: 0.053856 ms, speedup: 2.07
####################################################################################################
n: 1024, m: 8192
torch                          mean time: 0.211872 ms
norm_fp32                      mean time: 0.206912 ms, speedup: 1.02
norm_fp32x4                    mean time: 0.205552 ms, speedup: 1.03
norm_fp16                      mean time: 0.112928 ms, speedup: 1.88
norm_fp16x8_packed             mean time: 0.107008 ms, speedup: 1.98
####################################################################################################
n: 1024, m: 12800
torch                          mean time: 0.458320 ms
norm_fp32                      mean time: 0.350544 ms, speedup: 1.31
norm_fp32x4                    mean time: 0.344272 ms, speedup: 1.33
norm_fp16                      mean time: 0.184640 ms, speedup: 2.48
norm_fp16x8_packed             mean time: 0.164720 ms, speedup: 2.78
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.209008 ms
norm_fp32                      mean time: 0.207680 ms, speedup: 1.01
norm_fp32x4                    mean time: 0.206848 ms, speedup: 1.01
norm_fp16                      mean time: 0.108416 ms, speedup: 1.93
norm_fp16x8_packed             mean time: 0.105712 ms, speedup: 1.98
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.414272 ms
norm_fp32                      mean time: 0.411104 ms, speedup: 1.01
norm_fp32x4                    mean time: 0.405792 ms, speedup: 1.02
norm_fp16                      mean time: 0.211952 ms, speedup: 1.95
norm_fp16x8_packed             mean time: 0.207488 ms, speedup: 2.00
####################################################################################################
n: 4096, m: 8192
torch                          mean time: 0.820816 ms
norm_fp32                      mean time: 0.813520 ms, speedup: 1.01
norm_fp32x4                    mean time: 0.805072 ms, speedup: 1.02
norm_fp16                      mean time: 0.418432 ms, speedup: 1.96
norm_fp16x8_packed             mean time: 0.406032 ms, speedup: 2.02
####################################################################################################
n: 4096, m: 12800
torch                          mean time: 1.444640 ms
norm_fp32                      mean time: 1.353856 ms, speedup: 1.07
norm_fp32x4                    mean time: 1.358960 ms, speedup: 1.06
norm_fp16                      mean time: 0.691024 ms, speedup: 2.09
norm_fp16x8_packed             mean time: 0.673696 ms, speedup: 2.14
```
