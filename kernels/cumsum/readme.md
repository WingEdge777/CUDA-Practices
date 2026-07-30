# cumsum

## Overview

cumsum kernels.

- [x] naive Torch cumsum
- [x] cumsum — FP32
- [x] cumsum — FP32x4
- [x] cumsum — BF16
- [x] cumsum — BF16x8 packed
- [x] cumsum — FP32x4 split-k
- [x] pytorch op bindings && diff check

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### Sample output

```bash
####################################################################################################
n: 1, m: 2048
torch                          mean time: 0.010736 ms
cumsum_fp32                    mean time: 0.011232 ms, speedup: 0.96
cumsum_fp32x4                  mean time: 0.007328 ms, speedup: 1.47
cumsum_fp32x4_split_k          mean time: 0.011072 ms, speedup: 0.97
cumsum_bf16                    mean time: 0.010864 ms, speedup: 0.99
cumsum_bf16x8_packed           mean time: 0.006880 ms, speedup: 1.56
####################################################################################################
n: 1, m: 4096
torch                          mean time: 0.010640 ms
cumsum_fp32                    mean time: 0.015248 ms, speedup: 0.70
cumsum_fp32x4                  mean time: 0.008832 ms, speedup: 1.20
cumsum_fp32x4_split_k          mean time: 0.011024 ms, speedup: 0.97
cumsum_bf16                    mean time: 0.015104 ms, speedup: 0.70
cumsum_bf16x8_packed           mean time: 0.007264 ms, speedup: 1.46
####################################################################################################
n: 1, m: 8192
torch                          mean time: 0.010704 ms
cumsum_fp32                    mean time: 0.025440 ms, speedup: 0.42
cumsum_fp32x4                  mean time: 0.011008 ms, speedup: 0.97
cumsum_fp32x4_split_k          mean time: 0.010976 ms, speedup: 0.98
cumsum_bf16                    mean time: 0.023568 ms, speedup: 0.45
cumsum_bf16x8_packed           mean time: 0.008896 ms, speedup: 1.20
####################################################################################################
n: 1, m: 12800
torch                          mean time: 0.011264 ms
cumsum_fp32                    mean time: 0.036640 ms, speedup: 0.31
cumsum_fp32x4                  mean time: 0.014784 ms, speedup: 0.76
cumsum_fp32x4_split_k          mean time: 0.011088 ms, speedup: 1.02
cumsum_bf16                    mean time: 0.034208 ms, speedup: 0.33
cumsum_bf16x8_packed           mean time: 0.010928 ms, speedup: 1.03
####################################################################################################
n: 1, m: 32768
torch                          mean time: 0.011632 ms
cumsum_fp32                    mean time: 0.083344 ms, speedup: 0.14
cumsum_fp32x4                  mean time: 0.027552 ms, speedup: 0.42
cumsum_fp32x4_split_k          mean time: 0.011216 ms, speedup: 1.04
cumsum_bf16                    mean time: 0.079200 ms, speedup: 0.15
cumsum_bf16x8_packed           mean time: 0.017344 ms, speedup: 0.67
####################################################################################################
n: 1, m: 65536
torch                          mean time: 0.011568 ms
cumsum_fp32                    mean time: 0.170720 ms, speedup: 0.07
cumsum_fp32x4                  mean time: 0.054000 ms, speedup: 0.21
cumsum_fp32x4_split_k          mean time: 0.013232 ms, speedup: 0.87
cumsum_bf16                    mean time: 0.170656 ms, speedup: 0.07
cumsum_bf16x8_packed           mean time: 0.031840 ms, speedup: 0.36
####################################################################################################
n: 32, m: 2048
torch                          mean time: 0.017552 ms
cumsum_fp32                    mean time: 0.011216 ms, speedup: 1.56
cumsum_fp32x4                  mean time: 0.008928 ms, speedup: 1.97
cumsum_fp32x4_split_k          mean time: 0.012672 ms, speedup: 1.39
cumsum_bf16                    mean time: 0.011296 ms, speedup: 1.55
cumsum_bf16x8_packed           mean time: 0.007472 ms, speedup: 2.35
####################################################################################################
n: 32, m: 4096
torch                          mean time: 0.017648 ms
cumsum_fp32                    mean time: 0.017120 ms, speedup: 1.03
cumsum_fp32x4                  mean time: 0.011072 ms, speedup: 1.59
cumsum_fp32x4_split_k          mean time: 0.013136 ms, speedup: 1.34
cumsum_bf16                    mean time: 0.015456 ms, speedup: 1.14
cumsum_bf16x8_packed           mean time: 0.008960 ms, speedup: 1.97
####################################################################################################
n: 32, m: 8192
torch                          mean time: 0.029056 ms
cumsum_fp32                    mean time: 0.027776 ms, speedup: 1.05
cumsum_fp32x4                  mean time: 0.016704 ms, speedup: 1.74
cumsum_fp32x4_split_k          mean time: 0.015472 ms, speedup: 1.88
cumsum_bf16                    mean time: 0.025312 ms, speedup: 1.15
cumsum_bf16x8_packed           mean time: 0.011056 ms, speedup: 2.63
####################################################################################################
n: 32, m: 12800
torch                          mean time: 0.029488 ms
cumsum_fp32                    mean time: 0.041856 ms, speedup: 0.70
cumsum_fp32x4                  mean time: 0.021568 ms, speedup: 1.37
cumsum_fp32x4_split_k          mean time: 0.017856 ms, speedup: 1.65
cumsum_bf16                    mean time: 0.040512 ms, speedup: 0.73
cumsum_bf16x8_packed           mean time: 0.014928 ms, speedup: 1.98
####################################################################################################
n: 32, m: 32768
torch                          mean time: 0.062624 ms
cumsum_fp32                    mean time: 0.097104 ms, speedup: 0.64
cumsum_fp32x4                  mean time: 0.046016 ms, speedup: 1.36
cumsum_fp32x4_split_k          mean time: 0.034320 ms, speedup: 1.82
cumsum_bf16                    mean time: 0.085440 ms, speedup: 0.73
cumsum_bf16x8_packed           mean time: 0.026912 ms, speedup: 2.33
####################################################################################################
n: 32, m: 65536
torch                          mean time: 0.119216 ms
cumsum_fp32                    mean time: 0.189936 ms, speedup: 0.63
cumsum_fp32x4                  mean time: 0.082832 ms, speedup: 1.44
cumsum_fp32x4_split_k          mean time: 0.068176 ms, speedup: 1.75
cumsum_bf16                    mean time: 0.168432 ms, speedup: 0.71
cumsum_bf16x8_packed           mean time: 0.046464 ms, speedup: 2.57
####################################################################################################
n: 64, m: 2048
torch                          mean time: 0.017280 ms
cumsum_fp32                    mean time: 0.012992 ms, speedup: 1.33
cumsum_fp32x4                  mean time: 0.009328 ms, speedup: 1.85
cumsum_fp32x4_split_k          mean time: 0.013216 ms, speedup: 1.31
cumsum_bf16                    mean time: 0.011264 ms, speedup: 1.53
cumsum_bf16x8_packed           mean time: 0.008432 ms, speedup: 2.05
####################################################################################################
n: 64, m: 4096
torch                          mean time: 0.027200 ms
cumsum_fp32                    mean time: 0.019216 ms, speedup: 1.42
cumsum_fp32x4                  mean time: 0.012928 ms, speedup: 2.10
cumsum_fp32x4_split_k          mean time: 0.015424 ms, speedup: 1.76
cumsum_bf16                    mean time: 0.017120 ms, speedup: 1.59
cumsum_bf16x8_packed           mean time: 0.009936 ms, speedup: 2.74
####################################################################################################
n: 64, m: 8192
torch                          mean time: 0.033648 ms
cumsum_fp32                    mean time: 0.031584 ms, speedup: 1.07
cumsum_fp32x4                  mean time: 0.018224 ms, speedup: 1.85
cumsum_fp32x4_split_k          mean time: 0.019664 ms, speedup: 1.71
cumsum_bf16                    mean time: 0.029264 ms, speedup: 1.15
cumsum_bf16x8_packed           mean time: 0.013056 ms, speedup: 2.58
####################################################################################################
n: 64, m: 12800
torch                          mean time: 0.048800 ms
cumsum_fp32                    mean time: 0.047040 ms, speedup: 1.04
cumsum_fp32x4                  mean time: 0.025200 ms, speedup: 1.94
cumsum_fp32x4_split_k          mean time: 0.027408 ms, speedup: 1.78
cumsum_bf16                    mean time: 0.041488 ms, speedup: 1.18
cumsum_bf16x8_packed           mean time: 0.017024 ms, speedup: 2.87
####################################################################################################
n: 64, m: 32768
torch                          mean time: 0.084032 ms
cumsum_fp32                    mean time: 0.112944 ms, speedup: 0.74
cumsum_fp32x4                  mean time: 0.061456 ms, speedup: 1.37
cumsum_fp32x4_split_k          mean time: 0.066016 ms, speedup: 1.27
cumsum_bf16                    mean time: 0.099024 ms, speedup: 0.85
cumsum_bf16x8_packed           mean time: 0.029856 ms, speedup: 2.81
####################################################################################################
n: 64, m: 65536
torch                          mean time: 0.155808 ms
cumsum_fp32                    mean time: 0.206000 ms, speedup: 0.76
cumsum_fp32x4                  mean time: 0.151776 ms, speedup: 1.03
cumsum_fp32x4_split_k          mean time: 0.161792 ms, speedup: 0.96
cumsum_bf16                    mean time: 0.198624 ms, speedup: 0.78
cumsum_bf16x8_packed           mean time: 0.060272 ms, speedup: 2.59
####################################################################################################
n: 128, m: 2048
torch                          mean time: 0.027488 ms
cumsum_fp32                    mean time: 0.015424 ms, speedup: 1.78
cumsum_fp32x4                  mean time: 0.010864 ms, speedup: 2.53
cumsum_fp32x4_split_k          mean time: 0.015520 ms, speedup: 1.77
cumsum_bf16                    mean time: 0.013248 ms, speedup: 2.07
cumsum_bf16x8_packed           mean time: 0.008992 ms, speedup: 3.06
####################################################################################################
n: 128, m: 4096
torch                          mean time: 0.032272 ms
cumsum_fp32                    mean time: 0.025376 ms, speedup: 1.27
cumsum_fp32x4                  mean time: 0.013344 ms, speedup: 2.42
cumsum_fp32x4_split_k          mean time: 0.019728 ms, speedup: 1.64
cumsum_bf16                    mean time: 0.021088 ms, speedup: 1.53
cumsum_bf16x8_packed           mean time: 0.010848 ms, speedup: 2.97
####################################################################################################
n: 128, m: 8192
torch                          mean time: 0.059312 ms
cumsum_fp32                    mean time: 0.044784 ms, speedup: 1.32
cumsum_fp32x4                  mean time: 0.023312 ms, speedup: 2.54
cumsum_fp32x4_split_k          mean time: 0.033728 ms, speedup: 1.76
cumsum_bf16                    mean time: 0.035824 ms, speedup: 1.66
cumsum_bf16x8_packed           mean time: 0.013712 ms, speedup: 4.33
####################################################################################################
n: 128, m: 12800
torch                          mean time: 0.065776 ms
cumsum_fp32                    mean time: 0.064288 ms, speedup: 1.02
cumsum_fp32x4                  mean time: 0.048144 ms, speedup: 1.37
cumsum_fp32x4_split_k          mean time: 0.053936 ms, speedup: 1.22
cumsum_bf16                    mean time: 0.052512 ms, speedup: 1.25
cumsum_bf16x8_packed           mean time: 0.021616 ms, speedup: 3.04
####################################################################################################
n: 128, m: 32768
torch                          mean time: 0.155696 ms
cumsum_fp32                    mean time: 0.143792 ms, speedup: 1.08
cumsum_fp32x4                  mean time: 0.108944 ms, speedup: 1.43
cumsum_fp32x4_split_k          mean time: 0.130896 ms, speedup: 1.19
cumsum_bf16                    mean time: 0.109408 ms, speedup: 1.42
cumsum_bf16x8_packed           mean time: 0.054048 ms, speedup: 2.88
####################################################################################################
n: 128, m: 65536
torch                          mean time: 0.272624 ms
cumsum_fp32                    mean time: 0.268912 ms, speedup: 1.01
cumsum_fp32x4                  mean time: 0.229376 ms, speedup: 1.19
cumsum_fp32x4_split_k          mean time: 0.415344 ms, speedup: 0.66
cumsum_bf16                    mean time: 0.204480 ms, speedup: 1.33
cumsum_bf16x8_packed           mean time: 0.114144 ms, speedup: 2.39
```
