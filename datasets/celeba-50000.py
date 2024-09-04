import os
import shutil
import random
from torchvision.datasets import CelebA

# CelebA µ¥ÀÌÅÍ¼ÂÀÇ ·çÆ® µð·ºÅä¸® ¼³Á¤
root = '/srv1/diffusion_dataset'
root1 = '/srv1/diffusion_dataset/celeba'
output_dir = '/srv1/dayena7/celeba-50000/'

# µ¥ÀÌÅÍ¼Â ·Îµå
celeba = CelebA(root=root, split='train', download=True)

# ¹«ÀÛÀ§·Î 50,000ÀåÀÇ ÀÎµ¦½º ¼±ÅÃ
indices = random.sample(range(len(celeba)), 50000)

# µð·ºÅä¸® »ý¼º
os.makedirs(output_dir, exist_ok=True)

# ¼±ÅÃµÈ ÀÌ¹ÌÁö¸¦ »õ·Î¿î µð·ºÅä¸®·Î º¹»ç
for idx in indices:
    img_filename = celeba.filename[idx]  # ÀÌ¹ÌÁö ÆÄÀÏ ÀÌ¸§ °¡Á®¿À±â
    img_path = os.path.join(root1, "img_align_celeba", img_filename)  # ÀüÃ¼ °æ·Î ±¸¼º
    shutil.copy(img_path, output_dir)