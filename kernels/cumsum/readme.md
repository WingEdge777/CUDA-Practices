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
torch                          mean time: 0.011104 ms
cumsum_fp32                    mean time: 0.011648 ms, speedup: 0.95
cumsum_fp32x4                  mean time: 0.007904 ms, speedup: 1.40
cumsum_bf16                    mean time: 0.011328 ms, speedup: 0.98
cumsum_bf16x8_packed           mean time: 0.007136 ms, speedup: 1.56
####################################################################################################
n: 1, m: 4096
torch                          mean time: 0.011360 ms
cumsum_fp32                    mean time: 0.017328 ms, speedup: 0.66
cumsum_fp32x4                  mean time: 0.009456 ms, speedup: 1.20
cumsum_bf16                    mean time: 0.017104 ms, speedup: 0.66
cumsum_bf16x8_packed           mean time: 0.008128 ms, speedup: 1.40
####################################################################################################
n: 1, m: 8192
torch                          mean time: 0.011280 ms
cumsum_fp32                    mean time: 0.027744 ms, speedup: 0.41
cumsum_fp32x4                  mean time: 0.012656 ms, speedup: 0.89
cumsum_bf16                    mean time: 0.025792 ms, speedup: 0.44
cumsum_bf16x8_packed           mean time: 0.009328 ms, speedup: 1.21
####################################################################################################
n: 1, m: 12800
torch                          mean time: 0.011584 ms
cumsum_fp32                    mean time: 0.039520 ms, speedup: 0.29
cumsum_fp32x4                  mean time: 0.016752 ms, speedup: 0.69
cumsum_bf16                    mean time: 0.038672 ms, speedup: 0.30
cumsum_bf16x8_packed           mean time: 0.012208 ms, speedup: 0.95
####################################################################################################
n: 1, m: 32768
torch                          mean time: 0.013168 ms
cumsum_fp32                    mean time: 0.104416 ms, speedup: 0.13
cumsum_fp32x4                  mean time: 0.037200 ms, speedup: 0.35
cumsum_bf16                    mean time: 0.117600 ms, speedup: 0.11
cumsum_bf16x8_packed           mean time: 0.023136 ms, speedup: 0.57
####################################################################################################
n: 1, m: 65536
torch                          mean time: 0.015632 ms
cumsum_fp32                    mean time: 0.205808 ms, speedup: 0.08
cumsum_fp32x4                  mean time: 0.066784 ms, speedup: 0.23
cumsum_bf16                    mean time: 0.187280 ms, speedup: 0.08
cumsum_bf16x8_packed           mean time: 0.033760 ms, speedup: 0.46
####################################################################################################
n: 32, m: 2048
torch                          mean time: 0.019168 ms
cumsum_fp32                    mean time: 0.013472 ms, speedup: 1.42
cumsum_fp32x4                  mean time: 0.009344 ms, speedup: 2.05
cumsum_bf16                    mean time: 0.013104 ms, speedup: 1.46
cumsum_bf16x8_packed           mean time: 0.008080 ms, speedup: 2.37
####################################################################################################
n: 32, m: 4096
torch                          mean time: 0.019472 ms
cumsum_fp32                    mean time: 0.019392 ms, speedup: 1.00
cumsum_fp32x4                  mean time: 0.011344 ms, speedup: 1.72
cumsum_bf16                    mean time: 0.017600 ms, speedup: 1.11
cumsum_bf16x8_packed           mean time: 0.009248 ms, speedup: 2.11
####################################################################################################
n: 32, m: 8192
torch                          mean time: 0.030096 ms
cumsum_fp32                    mean time: 0.029248 ms, speedup: 1.03
cumsum_fp32x4                  mean time: 0.016992 ms, speedup: 1.77
cumsum_bf16                    mean time: 0.028112 ms, speedup: 1.07
cumsum_bf16x8_packed           mean time: 0.011280 ms, speedup: 2.67
####################################################################################################
n: 32, m: 12800
torch                          mean time: 0.031664 ms
cumsum_fp32                    mean time: 0.042736 ms, speedup: 0.74
cumsum_fp32x4                  mean time: 0.022912 ms, speedup: 1.38
cumsum_bf16                    mean time: 0.042224 ms, speedup: 0.75
cumsum_bf16x8_packed           mean time: 0.016032 ms, speedup: 1.98
####################################################################################################
n: 32, m: 32768
torch                          mean time: 0.070128 ms
cumsum_fp32                    mean time: 0.114016 ms, speedup: 0.62
cumsum_fp32x4                  mean time: 0.048352 ms, speedup: 1.45
cumsum_bf16                    mean time: 0.109024 ms, speedup: 0.64
cumsum_bf16x8_packed           mean time: 0.028352 ms, speedup: 2.47
####################################################################################################
n: 32, m: 65536
torch                          mean time: 0.154960 ms
cumsum_fp32                    mean time: 0.222048 ms, speedup: 0.70
cumsum_fp32x4                  mean time: 0.087504 ms, speedup: 1.77
cumsum_bf16                    mean time: 0.185632 ms, speedup: 0.83
cumsum_bf16x8_packed           mean time: 0.048864 ms, speedup: 3.17
####################################################################################################
n: 64, m: 2048
torch                          mean time: 0.018032 ms
cumsum_fp32                    mean time: 0.013408 ms, speedup: 1.34
cumsum_fp32x4                  mean time: 0.010576 ms, speedup: 1.70
cumsum_bf16                    mean time: 0.013040 ms, speedup: 1.38
cumsum_bf16x8_packed           mean time: 0.008880 ms, speedup: 2.03
####################################################################################################
n: 64, m: 4096
torch                          mean time: 0.029600 ms
cumsum_fp32                    mean time: 0.019728 ms, speedup: 1.50
cumsum_fp32x4                  mean time: 0.013136 ms, speedup: 2.25
cumsum_bf16                    mean time: 0.018864 ms, speedup: 1.57
cumsum_bf16x8_packed           mean time: 0.010608 ms, speedup: 2.79
####################################################################################################
n: 64, m: 8192
torch                          mean time: 0.036432 ms
cumsum_fp32                    mean time: 0.032128 ms, speedup: 1.13
cumsum_fp32x4                  mean time: 0.019024 ms, speedup: 1.92
cumsum_bf16                    mean time: 0.030384 ms, speedup: 1.20
cumsum_bf16x8_packed           mean time: 0.013728 ms, speedup: 2.65
####################################################################################################
n: 64, m: 12800
torch                          mean time: 0.053328 ms
cumsum_fp32                    mean time: 0.051024 ms, speedup: 1.05
cumsum_fp32x4                  mean time: 0.025696 ms, speedup: 2.08
cumsum_bf16                    mean time: 0.045024 ms, speedup: 1.18
cumsum_bf16x8_packed           mean time: 0.017344 ms, speedup: 3.07
####################################################################################################
n: 64, m: 32768
torch                          mean time: 0.104320 ms
cumsum_fp32                    mean time: 0.132784 ms, speedup: 0.79
cumsum_fp32x4                  mean time: 0.068496 ms, speedup: 1.52
cumsum_bf16                    mean time: 0.117776 ms, speedup: 0.89
cumsum_bf16x8_packed           mean time: 0.035392 ms, speedup: 2.95
####################################################################################################
n: 64, m: 65536
torch                          mean time: 0.194032 ms
cumsum_fp32                    mean time: 0.236752 ms, speedup: 0.82
cumsum_fp32x4                  mean time: 0.130464 ms, speedup: 1.49
cumsum_bf16                    mean time: 0.211936 ms, speedup: 0.92
cumsum_bf16x8_packed           mean time: 0.064704 ms, speedup: 3.00
####################################################################################################
n: 128, m: 2048
torch                          mean time: 0.029296 ms
cumsum_fp32                    mean time: 0.016288 ms, speedup: 1.80
cumsum_fp32x4                  mean time: 0.010912 ms, speedup: 2.68
cumsum_bf16                    mean time: 0.013952 ms, speedup: 2.10
cumsum_bf16x8_packed           mean time: 0.009488 ms, speedup: 3.09
####################################################################################################
n: 128, m: 4096
torch                          mean time: 0.034208 ms
cumsum_fp32                    mean time: 0.026496 ms, speedup: 1.29
cumsum_fp32x4                  mean time: 0.014656 ms, speedup: 2.33
cumsum_bf16                    mean time: 0.021376 ms, speedup: 1.60
cumsum_bf16x8_packed           mean time: 0.011568 ms, speedup: 2.96
####################################################################################################
n: 128, m: 8192
torch                          mean time: 0.060352 ms
cumsum_fp32                    mean time: 0.045216 ms, speedup: 1.33
cumsum_fp32x4                  mean time: 0.024064 ms, speedup: 2.51
cumsum_bf16                    mean time: 0.037856 ms, speedup: 1.59
cumsum_bf16x8_packed           mean time: 0.015056 ms, speedup: 4.01
####################################################################################################
n: 128, m: 12800
torch                          mean time: 0.072608 ms
cumsum_fp32                    mean time: 0.067568 ms, speedup: 1.07
cumsum_fp32x4                  mean time: 0.049152 ms, speedup: 1.48
cumsum_bf16                    mean time: 0.062512 ms, speedup: 1.16
cumsum_bf16x8_packed           mean time: 0.023616 ms, speedup: 3.07
####################################################################################################
n: 128, m: 32768
torch                          mean time: 0.170880 ms
cumsum_fp32                    mean time: 0.152416 ms, speedup: 1.12
cumsum_fp32x4                  mean time: 0.114880 ms, speedup: 1.49
cumsum_bf16                    mean time: 0.130800 ms, speedup: 1.31
cumsum_bf16x8_packed           mean time: 0.055840 ms, speedup: 3.06
####################################################################################################
n: 128, m: 65536
torch                          mean time: 0.291072 ms
cumsum_fp32                    mean time: 0.271328 ms, speedup: 1.07
cumsum_fp32x4                  mean time: 0.223520 ms, speedup: 1.30
cumsum_bf16                    mean time: 0.215072 ms, speedup: 1.35
cumsum_bf16x8_packed           mean time: 0.110640 ms, speedup: 2.63
```
