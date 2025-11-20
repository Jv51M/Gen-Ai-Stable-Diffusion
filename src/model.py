import typing
from diffusers import DiffusionPipeline
import torch

def create_image(
        model:str,
        prompt: str,
        num_inference: int,
        neg_prompt: str,
        options: list = [],
        guidance: float = 0.7,
        comp_device:str = "cpu") -> str :

    for opt in options:
        prompt = f"{prompt}, {opt}"
    if model == 'diffusion':
        model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    d_type = torch.float32 if comp_device=='cpu' else torch.float16

    pipe = DiffusionPipeline.from_pretrained(model,dtype=d_type)
    pipe.to(comp_device)
    image = pipe(prompt,num_inference_steps=num_inference,negative_prompt=neg_prompt,guidance_scale=guidance).images[0]
    filename=prompt.replace(' ','_')+".png"
    image.save(filename)
    return filename

if __name__ == '__main__':
    create_image('diffusion',
                 'A futuristic city in the clouds, photorealistic',
                 20,
                 'cartoon',
                 [],
                 6.0,'cpu')
