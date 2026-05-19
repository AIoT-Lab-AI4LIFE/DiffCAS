import os
import shutil

# 1. Khai báo đường dẫn
source_base = "/home/nvidia-lab/ai4life/kienpt/dataset_remove_background_3_channel_png_for_attmap"
target_base = "/home/nvidia-lab/ai4life/pytorch-cyclegan-and-pix2pix/datasets/my_attmap_dataset"

# Các thư mục cha
modes = ["train", "val", "test"]
# Các thư mục con
folders = ["A", "B"]

print(f"--- Bắt đầu copy và xử lý dữ liệu sang: {target_base} ---")

for mode in modes:
    for folder in folders:
        # Đường dẫn nguồn: .../train/A
        src_path = os.path.join(source_base, mode, folder)
        # Đường dẫn đích: .../my_attmap_dataset/train/A
        dst_path = os.path.join(target_base, mode, folder)
        
        if os.path.exists(src_path):
            os.makedirs(dst_path, exist_ok=True)
            print(f"Đang xử lý {mode}/{folder}...")
            
            for filename in os.listdir(src_path):
                file_src = os.path.join(src_path, filename)
                
                # Nếu là folder B thì đổi tên file AC -> NC
                if folder == "B":
                    new_filename = filename.replace("AC", "NC")
                else:
                    new_filename = filename
                
                file_dst = os.path.join(dst_path, new_filename)
                
                # Thực hiện copy
                if os.path.isfile(file_src):
                    shutil.copy2(file_src, file_dst)
        else:
            print(f"Bỏ qua: {mode}/{folder} (không tìm thấy nguồn)")

print("---")
print("Đã hoàn thành! Cấu trúc thư mục đích hiện tại:")
# Lệnh này để bạn nhìn thấy cây thư mục sau khi xong
os.system(f"ls -R {target_base} | grep ':$' | sed -e 's/:$//' -e 's/[^-][^\/]*\//--/g' -e 's/^/   /'")