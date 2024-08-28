from diffusers import StableDiffusionPipeline, DDIMScheduler

ddim = DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=ddim)

image = pipeline("A beautiful pretty girl, having a black hair, is smoking. Her favorite color is pink, so she is wearing a pink neat. The grassland is a background. The butterflies fly around her.").images[0]

image.save("smoking_pink_diana.png")