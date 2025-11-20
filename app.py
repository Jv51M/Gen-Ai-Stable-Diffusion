import time
import typing
import torch
import streamlit as st
from diffusers import DiffusionPipeline
from PIL import PngImagePlugin, ImageDraw, ImageFont

#saving the image after adding a watermark and saving prompt to its metadata
def save_img(image,prompt:str):
    filename=f"{prompt[:20]}-{time.strftime('%Y%m%d-%H%M%S')}.png"
    watermark = 'Ai generated'
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype('arial.ttf', int(width/20))
    except IOError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), watermark, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x, y = width-text_width-10 , height-text_height-10
    draw.text((x,y),watermark,font=font,fill=(255,255,255,128))
    filename=filename.replace(' ','_')
    filename=filename.replace(',','')
    #adding metadata
    meta_info= PngImagePlugin.PngInfo()
    meta_info.add_text('prompt',prompt)

    image.save(f'assets/{filename}',pnginfo=meta_info,format='PNG')

# Passing parameters to generate the image
def create_image(
#        model:str,
        prompt: str,
        num_inference: int,
        neg_prompt: str,
        options: list,         # image generation enhancers selected from sidebar
        guidance: float = 5.0,
        comp_device:str = "cpu") -> str :

    for opt in options:
        prompt = f"{prompt}, {opt}"
    #assigning model parameters and generating  pipline
    model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    d_type = torch.float32 if comp_device=='cpu' else torch.float16
    pipe = DiffusionPipeline.from_pretrained(model,dtype=d_type)
    pipe.to(comp_device)
    #to keep track of the steps completed
    status= st.empty()
    def call_back(step, timestep, latents):
        status.write(f"🔄 Step {step}/{num_inference}")
    image = pipe(prompt,num_inference_steps=num_inference,negative_prompt=neg_prompt,guidance_scale=guidance,
                 callback=call_back,callback_steps=1).images[0]
    return image

st.title("Stable Diffusion v1.5")
if 'key' not in st.session_state:
    st.markdown('''### 🎨 Image Generation Controls
* **🖥️ Choose a Model:** Select your preferred model from the dropdown menu.
* **⚡ Performance Slider:** Adjust the slider to improve image quality. (Higher performance = more processing time.)
* **🎯 Accuracy / Guidance:** Control how closely the image matches your prompt.
* **🚫 Negative Prompts:** Remove unwanted features like `"Cartoon"`, `"Anime"`, or `"Water"`.''')
    if st.button('Continue'):
        st.session_state['key']=0
        st.session_state['messages']=[]
        st.rerun()
#model taking user parameter input for the model
else:
    st.session_state.key+=1   #key of every download button has to be unique, st.session_state.key is incremented everytime
    with st.sidebar:
        inference = st.slider(label='Performance',min_value=1,max_value=20,value=7)
        guide = st.slider(label='Prompt Similarity', min_value=0.0,max_value=10.0,value=7.5)
        optional_prompt=st.pills('Image Enhancement',['photorealistic', 'cinematic', 'lighting', '4k','sharp',
                                                    'vibrant', 'colors', 'depth of field', 'dramatic', 'shadows', 'realistic'],
                                 selection_mode='multi')
        neg_prompt= st.text_area('Enter Negative Prompt')
        comp_device= st.segmented_control('Select Device',['cpu','cuda'],default='cpu')

    # Unloading the previous image data and from the session_state
    def show_msgs():
        for message in st.session_state.messages:
            with st.chat_message('User'):
                st.write(message['prompt'])
            with st.chat_message('Assistant'):
                st.image(message['image'])
                st.button(label='download',
                          on_click=save_img,
                          args=(message['image'], message['prompt']),
                          key='download_button' + str(st.session_state.key))
    show_msgs()
    # structuring a chat format to display the prompt and the image generated
    prompt= st.chat_input()
    if prompt:
            with st.spinner('Generating image...'):
                try:
                    image=create_image(prompt,inference,neg_prompt,optional_prompt,guide,comp_device)
                    st.session_state.messages.append({"prompt": prompt, "image": image})
                    show_msgs()
                except Exception as e:
                    st.error(f'error generating image{e}')

