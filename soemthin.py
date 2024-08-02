from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify
import io

app = Flask(__name__)

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def process_images(images):
    processed_images = []
    
    for image in images:
        img = Image.open(io.BytesIO(image.read()))
        
        # Resize image to ensure consistency
        img = img.convert("RGB")
        img = img.resize((224, 224))  # Adjust size according to your model's requirement
        
        processed_images.append(img)
    
    try:
        # Process images using feature extractor
        pixel_values = feature_extractor(images=processed_images, return_tensors="pt", padding=True).pixel_values
        pixel_values = pixel_values.to(device)
        
        # Generate captions
        output_ids = model.generate(pixel_values, **gen_kwargs)
        captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        return jsonify(captions)
    
    except Exception as e:
        # Return a JSON response with error message
        print(str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    images = request.files.getlist('images')
    return process_images(images)

if __name__ == '__main__':
    app.run(debug=True)
