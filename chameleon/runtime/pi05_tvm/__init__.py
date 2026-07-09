"""pi05 TVM 运行时：复用 TRT vit 前缀嵌入，denoise 环换成 mlc-vla 的 M1 TVM engine
（expert-0 prefill 固化 prefix K/V + suffix-only denoise_step_kv，host Euler 去噪）。"""
