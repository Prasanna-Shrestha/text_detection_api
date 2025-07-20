from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch
from PIL import Image

def load_model():
    model_name = "microsoft/trocr-small-handwritten"  # smaller for lower RAM
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.eval()
    return model, processor

def predict_word(image: Image.Image, model, processor):
    # Resize for model input
    image = image.resize((224, 224))
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()
