import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Cargo el procesador y los modelos preentrenados
processors = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Carga una imagen (colocar direccion de la imagen)
img_path = "ChatGPT Image 27 sept 2025, 12_15_222.png"

# Comvierte a formato RGB
Image = Image.open(img_path).convert('RGB')

# No necesitas una pregunta para subtitular una imagen
text = "the image of"
inputs = processors(images=Image, text=text, return_tensors="pt")

# Generar un título para la imagen
outputs = model.generate(**inputs, max_length=50)

# Decode the generated tokens to text
caption = processors.decode(outputs[0], skip_special_tokens=True)

# Imprimir el título
print(caption)