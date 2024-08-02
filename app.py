from flask import Flask, render_template, request, jsonify
from your_module import process_images  # Ensure you have the correct import

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # Access uploaded images from the request
    images = request.files.getlist('images')
    
    # Process images and generate captions
    try:
        captions = process_images(images)
        return jsonify(captions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
