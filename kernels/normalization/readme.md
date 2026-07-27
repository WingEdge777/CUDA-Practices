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
torch                          mean time: 0.008720 ms
norm_fp32                      mean time: 0.010832 ms, speedup: 0.81
norm_fp32x4                    mean time: 0.009280 ms, speedup: 0.94
####################################################################################################
n: 64, m: 4096
torch                          mean time: 0.009456 ms
norm_fp32                      mean time: 0.012672 ms, speedup: 0.75
norm_fp32x4                    mean time: 0.013792 ms, speedup: 0.69
####################################################################################################
n: 64, m: 8192
torch                          mean time: 0.013664 ms
norm_fp32                      mean time: 0.022016 ms, speedup: 0.62
norm_fp32x4                    mean time: 0.019456 ms, speedup: 0.70
####################################################################################################
n: 64, m: 12800
torch                          mean time: 0.018128 ms
norm_fp32                      mean time: 0.032112 ms, speedup: 0.56
norm_fp32x4                    mean time: 0.029264 ms, speedup: 0.62
####################################################################################################
n: 128, m: 2048
torch                          mean time: 0.011808 ms
norm_fp32                      mean time: 0.013376 ms, speedup: 0.88
norm_fp32x4                    mean time: 0.013376 ms, speedup: 0.88
####################################################################################################
n: 128, m: 4096
torch                          mean time: 0.013936 ms
norm_fp32                      mean time: 0.019712 ms, speedup: 0.71
norm_fp32x4                    mean time: 0.017696 ms, speedup: 0.79
####################################################################################################
n: 128, m: 8192
torch                          mean time: 0.021920 ms
norm_fp32                      mean time: 0.038336 ms, speedup: 0.57
norm_fp32x4                    mean time: 0.036128 ms, speedup: 0.61
####################################################################################################
n: 128, m: 12800
torch                          mean time: 0.052016 ms
norm_fp32                      mean time: 0.050432 ms, speedup: 1.03
norm_fp32x4                    mean time: 0.048064 ms, speedup: 1.08
####################################################################################################
n: 512, m: 2048
torch                          mean time: 0.017472 ms
norm_fp32                      mean time: 0.029904 ms, speedup: 0.58
norm_fp32x4                    mean time: 0.027104 ms, speedup: 0.64
####################################################################################################
n: 512, m: 4096
torch                          mean time: 0.046032 ms
norm_fp32                      mean time: 0.056512 ms, speedup: 0.81
norm_fp32x4                    mean time: 0.052528 ms, speedup: 0.88
####################################################################################################
n: 512, m: 8192
torch                          mean time: 0.104880 ms
norm_fp32                      mean time: 0.107616 ms, speedup: 0.97
norm_fp32x4                    mean time: 0.103616 ms, speedup: 1.01
####################################################################################################
n: 512, m: 12800
torch                          mean time: 0.184000 ms
norm_fp32                      mean time: 0.169296 ms, speedup: 1.09
norm_fp32x4                    mean time: 0.167936 ms, speedup: 1.10
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.048208 ms
norm_fp32                      mean time: 0.055712 ms, speedup: 0.87
norm_fp32x4                    mean time: 0.052384 ms, speedup: 0.92
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.106976 ms
norm_fp32                      mean time: 0.106160 ms, speedup: 1.01
norm_fp32x4                    mean time: 0.104064 ms, speedup: 1.03
####################################################################################################
n: 1024, m: 8192
torch                          mean time: 0.214240 ms
norm_fp32                      mean time: 0.208224 ms, speedup: 1.03
norm_fp32x4                    mean time: 0.207440 ms, speedup: 1.03
####################################################################################################
n: 1024, m: 12800
torch                          mean time: 0.363344 ms
norm_fp32                      mean time: 0.327584 ms, speedup: 1.11
norm_fp32x4                    mean time: 0.323584 ms, speedup: 1.12
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.208816 ms
norm_fp32                      mean time: 0.218960 ms, speedup: 0.95
norm_fp32x4                    mean time: 0.207184 ms, speedup: 1.01
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.414640 ms
norm_fp32                      mean time: 0.410368 ms, speedup: 1.01
norm_fp32x4                    mean time: 0.405984 ms, speedup: 1.02
####################################################################################################
n: 4096, m: 8192
torch                          mean time: 0.823744 ms
norm_fp32                      mean time: 0.817600 ms, speedup: 1.01
norm_fp32x4                    mean time: 0.809456 ms, speedup: 1.02
####################################################################################################
n: 4096, m: 12800
torch                          mean time: 1.374640 ms
norm_fp32                      mean time: 1.288704 ms, speedup: 1.07
norm_fp32x4                    mean time: 1.288896 ms, speedup: 1.07
```
