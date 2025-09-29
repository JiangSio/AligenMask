import cv2
import numpy as np

# 1. 读取原图和掩码（确保尺寸一致）
image = cv2.imread('004.png')    # 原图（彩色）
mask = cv2.imread('004_foreground.png', cv2.IMREAD_GRAYSCALE)  # 掩码（单通道）

# 3. 归一化掩码到 [0, 1]
mask_normalized = mask.astype(np.float32) / 255.0

mask_normalized = cv2.merge([mask_normalized, mask_normalized, mask_normalized])
mask_normalized = 1-mask_normalized
# 5. 相乘
result = cv2.multiply(image.astype(np.float32), mask_normalized)

# 6. 转换为8位并保存
result = np.clip(result, 0, 255).astype(np.uint8)
cv2.imwrite('result1.png', result)

width, height = 1024, 1024
white_image = np.ones((height, width, 3), dtype=np.uint8) * 255

# 保存图像
cv2.imwrite('white_image.png', white_image)