import os
from diffusers import DDIMPipeline

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddim = DDIMPipeline.from_pretrained(model_id)

num_images = 50000

output_dir = "/home/parkjoe/PycharmProjects/ddim/image_samples/ddpm-cifar10_ddim-50000"
os.makedirs(output_dir, exist_ok=True)

for i in range(num_images):
    # run pipeline in inference (sample random noise and denoise)
    image = ddim(num_inference_steps=100).images[0]

    image.save(os.path.join(output_dir, f"{i}.png"))