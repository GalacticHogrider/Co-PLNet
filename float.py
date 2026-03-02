import numpy as np

# float16 的精度演示
x = np.float16(10.123)
print(f"输入: 10.123, float16存储: {x}")  # 10.125 (已经丢失精度)

# 距离计算中的误差累积
a = np.array([10.123, 20.456], dtype=np.float16)
b = np.array([10.124, 20.457], dtype=np.float16)

dist_16 = np.sqrt(((a - b) ** 2).sum())
print(f"float16 距离: {dist_16}")  # 可能是 0.0 或异常值

# 同样计算用 float64
a_64 = np.array([10.123, 20.456], dtype=np.float64)
b_64 = np.array([10.124, 20.457], dtype=np.float64)
dist_64 = np.sqrt(((a_64 - b_64) ** 2).sum())
print(f"float64 距离: {dist_64}")  # 正确的小距离值