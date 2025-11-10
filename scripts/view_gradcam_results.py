import os
from PIL import Image
import matplotlib.pyplot as plt

# --- Cấu hình ---
cam_dir = r"runs\rgb_resnet18_gpu\cam_report"  # Thư mục chứa kết quả Grad-CAM
per_row = 3  # số ảnh mỗi hàng

# --- Lấy danh sách ảnh PNG ---
images = [f for f in os.listdir(cam_dir) if f.lower().endswith(".png") and "poster" not in f]
images.sort()

if not images:
    print("❌ Không tìm thấy ảnh Grad-CAM trong thư mục:", cam_dir)
    exit()

# --- Hiển thị ảnh ---
n = len(images)
rows = (n + per_row - 1) // per_row

plt.figure(figsize=(per_row * 5, rows * 5))
for i, img_file in enumerate(images):
    img_path = os.path.join(cam_dir, img_file)
    img = Image.open(img_path)

    plt.subplot(rows, per_row, i + 1)
    plt.imshow(img)
    plt.axis("off")

    # Gợi ý tên lớp từ tên file
    title = os.path.splitext(img_file)[0]
    plt.title(title, fontsize=10)

plt.tight_layout()
plt.show()
