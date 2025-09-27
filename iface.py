import gradio as gr
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Cargo el procesador y el modelo
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    # Convierte la matriz numpy en una imagen PIL y la convierte a RGB
    raw_image = Image.fromarray(input_image).convert("RGB")

    # No necesitas una pregunta para subtitular una imagen
    text = "the image of"
    inputs = processor(images=raw_image, text=text, return_tensors="pt")

    # Generar un título para la imagen
    outputs = model.generate(**inputs, max_length=50)

    # Decodificar el resultado a texto
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

# Interfaz de Gradio
iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
)

if __name__ == "__main__":
    iface.launch()
