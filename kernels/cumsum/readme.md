# cumsum

## Overview

cumsum kernels.

- [x] naive Torch cumsum
- [x] cumsum — FP32
- [x] cumsum — FP32x4
- [x] cumsum — FP32x4 split-k
- [x] cumsum — FP32x4 multi-CTA scan
- [x] cumsum — BF16
- [x] cumsum — BF16x8 packed
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
torch                          mean time: 0.011200 ms
cumsum_fp32                    mean time: 0.011424 ms, speedup: 0.98
cumsum_fp32x4                  mean time: 0.007344 ms, speedup: 1.53
cumsum_fp32x4_split_k          mean time: 0.011232 ms, speedup: 1.00
cumsum_fp32x4_multi_cta_scan   mean time: 0.010640 ms, speedup: 1.05
cumsum_bf16                    mean time: 0.010960 ms, speedup: 1.02
cumsum_bf16x8_packed           mean time: 0.007024 ms, speedup: 1.59
####################################################################################################
n: 1, m: 4096
torch                          mean time: 0.009872 ms
cumsum_fp32                    mean time: 0.016352 ms, speedup: 0.60
cumsum_fp32x4                  mean time: 0.008960 ms, speedup: 1.10
cumsum_fp32x4_split_k          mean time: 0.011264 ms, speedup: 0.88
cumsum_fp32x4_multi_cta_scan   mean time: 0.010704 ms, speedup: 0.92
cumsum_bf16                    mean time: 0.015360 ms, speedup: 0.64
cumsum_bf16x8_packed           mean time: 0.007104 ms, speedup: 1.39
####################################################################################################
n: 1, m: 8192
torch                          mean time: 0.010816 ms
cumsum_fp32                    mean time: 0.026000 ms, speedup: 0.42
cumsum_fp32x4                  mean time: 0.011456 ms, speedup: 0.94
cumsum_fp32x4_split_k          mean time: 0.011136 ms, speedup: 0.97
cumsum_fp32x4_multi_cta_scan   mean time: 0.011264 ms, speedup: 0.96
cumsum_bf16                    mean time: 0.025040 ms, speedup: 0.43
cumsum_bf16x8_packed           mean time: 0.009072 ms, speedup: 1.19
####################################################################################################
n: 1, m: 12800
torch                          mean time: 0.011296 ms
cumsum_fp32                    mean time: 0.037904 ms, speedup: 0.30
cumsum_fp32x4                  mean time: 0.016272 ms, speedup: 0.69
cumsum_fp32x4_split_k          mean time: 0.011184 ms, speedup: 1.01
cumsum_fp32x4_multi_cta_scan   mean time: 0.011728 ms, speedup: 0.96
cumsum_bf16                    mean time: 0.035264 ms, speedup: 0.32
cumsum_bf16x8_packed           mean time: 0.011072 ms, speedup: 1.02
####################################################################################################
n: 1, m: 32768
torch                          mean time: 0.011392 ms
cumsum_fp32                    mean time: 0.087216 ms, speedup: 0.13
cumsum_fp32x4                  mean time: 0.029728 ms, speedup: 0.38
cumsum_fp32x4_split_k          mean time: 0.011600 ms, speedup: 0.98
cumsum_fp32x4_multi_cta_scan   mean time: 0.012560 ms, speedup: 0.91
cumsum_bf16                    mean time: 0.080576 ms, speedup: 0.14
cumsum_bf16x8_packed           mean time: 0.017600 ms, speedup: 0.65
####################################################################################################
n: 1, m: 65536
torch                          mean time: 0.012288 ms
cumsum_fp32                    mean time: 0.170208 ms, speedup: 0.07
cumsum_fp32x4                  mean time: 0.053696 ms, speedup: 0.23
cumsum_fp32x4_split_k          mean time: 0.012400 ms, speedup: 0.99
cumsum_fp32x4_multi_cta_scan   mean time: 0.014176 ms, speedup: 0.87
cumsum_bf16                    mean time: 0.156416 ms, speedup: 0.08
cumsum_bf16x8_packed           mean time: 0.030576 ms, speedup: 0.40
####################################################################################################
n: 32, m: 2048
torch                          mean time: 0.016768 ms
cumsum_fp32                    mean time: 0.012048 ms, speedup: 1.39
cumsum_fp32x4                  mean time: 0.009024 ms, speedup: 1.86
cumsum_fp32x4_split_k          mean time: 0.012240 ms, speedup: 1.37
cumsum_fp32x4_multi_cta_scan   mean time: 0.012048 ms, speedup: 1.39
cumsum_bf16                    mean time: 0.011152 ms, speedup: 1.50
cumsum_bf16x8_packed           mean time: 0.007152 ms, speedup: 2.34
####################################################################################################
n: 32, m: 4096
torch                          mean time: 0.017712 ms
cumsum_fp32                    mean time: 0.019920 ms, speedup: 0.89
cumsum_fp32x4                  mean time: 0.013072 ms, speedup: 1.35
cumsum_fp32x4_split_k          mean time: 0.013616 ms, speedup: 1.30
cumsum_fp32x4_multi_cta_scan   mean time: 0.012848 ms, speedup: 1.38
cumsum_bf16                    mean time: 0.017056 ms, speedup: 1.04
cumsum_bf16x8_packed           mean time: 0.008928 ms, speedup: 1.98
####################################################################################################
n: 32, m: 8192
torch                          mean time: 0.027904 ms
cumsum_fp32                    mean time: 0.027552 ms, speedup: 1.01
cumsum_fp32x4                  mean time: 0.016896 ms, speedup: 1.65
cumsum_fp32x4_split_k          mean time: 0.016240 ms, speedup: 1.72
cumsum_fp32x4_multi_cta_scan   mean time: 0.015408 ms, speedup: 1.81
cumsum_bf16                    mean time: 0.025520 ms, speedup: 1.09
cumsum_bf16x8_packed           mean time: 0.011344 ms, speedup: 2.46
####################################################################################################
n: 32, m: 12800
torch                          mean time: 0.029952 ms
cumsum_fp32                    mean time: 0.044016 ms, speedup: 0.68
cumsum_fp32x4                  mean time: 0.022608 ms, speedup: 1.32
cumsum_fp32x4_split_k          mean time: 0.019024 ms, speedup: 1.57
cumsum_fp32x4_multi_cta_scan   mean time: 0.018432 ms, speedup: 1.62
cumsum_bf16                    mean time: 0.037600 ms, speedup: 0.80
cumsum_bf16x8_packed           mean time: 0.015296 ms, speedup: 1.96
####################################################################################################
n: 32, m: 32768
torch                          mean time: 0.062928 ms
cumsum_fp32                    mean time: 0.097488 ms, speedup: 0.65
cumsum_fp32x4                  mean time: 0.046080 ms, speedup: 1.37
cumsum_fp32x4_split_k          mean time: 0.033904 ms, speedup: 1.86
cumsum_fp32x4_multi_cta_scan   mean time: 0.032112 ms, speedup: 1.96
cumsum_bf16                    mean time: 0.088256 ms, speedup: 0.71
cumsum_bf16x8_packed           mean time: 0.026800 ms, speedup: 2.35
####################################################################################################
n: 32, m: 65536
torch                          mean time: 0.118672 ms
cumsum_fp32                    mean time: 0.192416 ms, speedup: 0.62
cumsum_fp32x4                  mean time: 0.083872 ms, speedup: 1.41
cumsum_fp32x4_split_k          mean time: 0.081984 ms, speedup: 1.45
cumsum_fp32x4_multi_cta_scan   mean time: 0.057328 ms, speedup: 2.07
cumsum_bf16                    mean time: 0.170112 ms, speedup: 0.70
cumsum_bf16x8_packed           mean time: 0.046000 ms, speedup: 2.58
####################################################################################################
n: 64, m: 2048
torch                          mean time: 0.017520 ms
cumsum_fp32                    mean time: 0.013056 ms, speedup: 1.34
cumsum_fp32x4                  mean time: 0.010720 ms, speedup: 1.63
cumsum_fp32x4_split_k          mean time: 0.013632 ms, speedup: 1.29
cumsum_fp32x4_multi_cta_scan   mean time: 0.014064 ms, speedup: 1.25
cumsum_bf16                    mean time: 0.013104 ms, speedup: 1.34
cumsum_bf16x8_packed           mean time: 0.009088 ms, speedup: 1.93
####################################################################################################
n: 64, m: 4096
torch                          mean time: 0.030720 ms
cumsum_fp32                    mean time: 0.022976 ms, speedup: 1.34
cumsum_fp32x4                  mean time: 0.015360 ms, speedup: 2.00
cumsum_fp32x4_split_k          mean time: 0.017408 ms, speedup: 1.76
cumsum_fp32x4_multi_cta_scan   mean time: 0.016752 ms, speedup: 1.83
cumsum_bf16                    mean time: 0.019280 ms, speedup: 1.59
cumsum_bf16x8_packed           mean time: 0.010176 ms, speedup: 3.02
####################################################################################################
n: 64, m: 8192
torch                          mean time: 0.034640 ms
cumsum_fp32                    mean time: 0.033552 ms, speedup: 1.03
cumsum_fp32x4                  mean time: 0.019056 ms, speedup: 1.82
cumsum_fp32x4_split_k          mean time: 0.021168 ms, speedup: 1.64
cumsum_fp32x4_multi_cta_scan   mean time: 0.018992 ms, speedup: 1.82
cumsum_bf16                    mean time: 0.029728 ms, speedup: 1.17
cumsum_bf16x8_packed           mean time: 0.013088 ms, speedup: 2.65
####################################################################################################
n: 64, m: 12800
torch                          mean time: 0.049744 ms
cumsum_fp32                    mean time: 0.048144 ms, speedup: 1.03
cumsum_fp32x4                  mean time: 0.025280 ms, speedup: 1.97
cumsum_fp32x4_split_k          mean time: 0.027664 ms, speedup: 1.80
cumsum_fp32x4_multi_cta_scan   mean time: 0.026336 ms, speedup: 1.89
cumsum_bf16                    mean time: 0.042736 ms, speedup: 1.16
cumsum_bf16x8_packed           mean time: 0.017248 ms, speedup: 2.88
####################################################################################################
n: 64, m: 32768
torch                          mean time: 0.082576 ms
cumsum_fp32                    mean time: 0.112608 ms, speedup: 0.73
cumsum_fp32x4                  mean time: 0.060128 ms, speedup: 1.37
cumsum_fp32x4_split_k          mean time: 0.069472 ms, speedup: 1.19
cumsum_fp32x4_multi_cta_scan   mean time: 0.058928 ms, speedup: 1.40
cumsum_bf16                    mean time: 0.098864 ms, speedup: 0.84
cumsum_bf16x8_packed           mean time: 0.032800 ms, speedup: 2.52
####################################################################################################
n: 64, m: 65536
torch                          mean time: 0.158384 ms
cumsum_fp32                    mean time: 0.208768 ms, speedup: 0.76
cumsum_fp32x4                  mean time: 0.126544 ms, speedup: 1.25
cumsum_fp32x4_split_k          mean time: 0.132704 ms, speedup: 1.19
cumsum_fp32x4_multi_cta_scan   mean time: 0.118544 ms, speedup: 1.34
cumsum_bf16                    mean time: 0.180832 ms, speedup: 0.88
cumsum_bf16x8_packed           mean time: 0.064448 ms, speedup: 2.46
####################################################################################################
n: 128, m: 2048
torch                          mean time: 0.027280 ms
cumsum_fp32                    mean time: 0.016288 ms, speedup: 1.67
cumsum_fp32x4                  mean time: 0.010768 ms, speedup: 2.53
cumsum_fp32x4_split_k          mean time: 0.015712 ms, speedup: 1.74
cumsum_fp32x4_multi_cta_scan   mean time: 0.014928 ms, speedup: 1.83
cumsum_bf16                    mean time: 0.013264 ms, speedup: 2.06
cumsum_bf16x8_packed           mean time: 0.009040 ms, speedup: 3.02
####################################################################################################
n: 128, m: 4096
torch                          mean time: 0.031920 ms
cumsum_fp32                    mean time: 0.025344 ms, speedup: 1.26
cumsum_fp32x4                  mean time: 0.014272 ms, speedup: 2.24
cumsum_fp32x4_split_k          mean time: 0.021104 ms, speedup: 1.51
cumsum_fp32x4_multi_cta_scan   mean time: 0.017648 ms, speedup: 1.81
cumsum_bf16                    mean time: 0.021120 ms, speedup: 1.51
cumsum_bf16x8_packed           mean time: 0.010960 ms, speedup: 2.91
####################################################################################################
n: 128, m: 8192
torch                          mean time: 0.057840 ms
cumsum_fp32                    mean time: 0.043520 ms, speedup: 1.33
cumsum_fp32x4                  mean time: 0.023472 ms, speedup: 2.46
cumsum_fp32x4_split_k          mean time: 0.032864 ms, speedup: 1.76
cumsum_fp32x4_multi_cta_scan   mean time: 0.029792 ms, speedup: 1.94
cumsum_bf16                    mean time: 0.035424 ms, speedup: 1.63
cumsum_bf16x8_packed           mean time: 0.013888 ms, speedup: 4.16
####################################################################################################
n: 128, m: 12800
torch                          mean time: 0.063072 ms
cumsum_fp32                    mean time: 0.061488 ms, speedup: 1.03
cumsum_fp32x4                  mean time: 0.050432 ms, speedup: 1.25
cumsum_fp32x4_split_k          mean time: 0.051712 ms, speedup: 1.22
cumsum_fp32x4_multi_cta_scan   mean time: 0.053120 ms, speedup: 1.19
cumsum_bf16                    mean time: 0.049728 ms, speedup: 1.27
cumsum_bf16x8_packed           mean time: 0.021408 ms, speedup: 2.95
####################################################################################################
n: 128, m: 32768
torch                          mean time: 0.150864 ms
cumsum_fp32                    mean time: 0.143760 ms, speedup: 1.05
cumsum_fp32x4                  mean time: 0.111440 ms, speedup: 1.35
cumsum_fp32x4_split_k          mean time: 0.163328 ms, speedup: 0.92
cumsum_fp32x4_multi_cta_scan   mean time: 0.143552 ms, speedup: 1.05
cumsum_bf16                    mean time: 0.126128 ms, speedup: 1.20
cumsum_bf16x8_packed           mean time: 0.068528 ms, speedup: 2.20
####################################################################################################
n: 128, m: 65536
torch                          mean time: 0.317344 ms
cumsum_fp32                    mean time: 0.264864 ms, speedup: 1.20
cumsum_fp32x4                  mean time: 0.216048 ms, speedup: 1.47
cumsum_fp32x4_split_k          mean time: 0.398704 ms, speedup: 0.80
cumsum_fp32x4_multi_cta_scan   mean time: 0.236176 ms, speedup: 1.34
cumsum_bf16                    mean time: 0.205568 ms, speedup: 1.54
cumsum_bf16x8_packed           mean time: 0.123072 ms, speedup: 2.58
```
