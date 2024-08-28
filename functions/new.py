import hashlib

def calculate_md5(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

# ¿¹½Ã »ç¿ë
file_path = "/home/parkjoe/.cache/diffusion_models_converted/ema_diffusion_celeba_model/ckpt.pth"
md5_hash = calculate_md5(file_path)
print(f"MD5 Hash: {md5_hash}")
