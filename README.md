# Stable Diffusion v1.5 Image Generation App

This is a Streamlit web application that uses Hugging Face's Diffusers library to generate images from text prompts with the Stable Diffusion v1.5 model.

---

## Features

- Generate images from custom text prompts using the Stable Diffusion v1.5 model.

- Adjustable performance slider to trade quality vs speed (`num_inference_steps`).

- Adjust prompt similarity/guidance strength for fine control of image style.

- Enter negative prompts to exclude undesired features.

- Select multiple image enhancement options (e.g., photorealistic, vibrant).

- Images saved with a semi-transparent watermark.

- Saved images include the original prompt in PNG metadata.

- Images are saved to a dedicated folder within the project repo (`assets/`).

- Session state management to maintain generation history in chat format.

- Interactive UI sidebar and main page layout using Streamlit widgets.

---

## Installation

Make sure you have Python 3.8+ installed, then create and activate a virtual environment:

```bash
python -m venv .env source .env/bin/activate  # On Windows: .env\Scripts\activate`
```
Install required libraries:

```bash
`pip install torch streamlit diffusers Pillow transformers accelerate`
```
The `diffusers` package requires PyTorch; install the version matching your hardware and CUDA setup from the [official PyTorch site](https://pytorch.org/get-started/locally/).

---

## Usage

Run the Streamlit app:

```bash
`streamlit run app.py`
```
- On first load, you will see introductory image generation controls.

- Click **Continue** to access the full model parameter sidebar and prompt input area.

- Enter your text prompt and optionally add negative prompts and enhancements.

- Click enter or submit the prompt to generate the image.

- View generated images, prompts, and download watermarked images with prompt metadata.

- Generated images are stored in the `assets/` folder.

---

## Code Highlights

- **Image Generation:** Uses `DiffusionPipeline` from Hugging Face Diffusers to generate images.

- **Watermarking:** Adds "Ai generated" watermark at the bottom right using PIL's `ImageDraw`.

- **Metadata:** Embeds prompt text as metadata in PNG files using `PngImagePlugin`.

- **Session State:** Maintains prompt and image history for interactive chat-like UI.

- **Folder Management:** Ensures saved images go to a specified subfolder (`assets/`).

---

## License

This project is released under the MIT License.

---

## References

- [Hugging Face Diffusers Documentation](https://huggingface.co/docs/diffusers)

- [Streamlit Documentation](https://docs.streamlit.io/)

- [Pillow Documentation](https://pillow.readthedocs.io/en/stable/)

---
