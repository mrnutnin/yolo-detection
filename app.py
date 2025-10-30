from flask import Flask, request, jsonify, send_file, render_template_string
import torch
from PIL import Image
import numpy as np
import io
import webbrowser

app = Flask(__name__)

# -------------------------
# Config
# -------------------------
MODEL_PATH = "best.pt"         # วาง best.pt ไว้ในโฟลเดอร์เดียวกับ app.py
YOLOV5_REPO = "yolov5-master"  # โฟลเดอร์ YOLOv5 local
PORT = 5050                     # เปลี่ยนพอร์ตเพื่อไม่ชนกับโปรแกรมอื่น

# -------------------------
# Load YOLOv5 Model
# -------------------------
model = torch.hub.load(
    YOLOV5_REPO,
    'custom',
    path=MODEL_PATH,
    source='local'
)

# -------------------------
# HTML หน้าเว็บ upload
# -------------------------
HTML_PAGE = """
<!doctype html>
<html>
<head>
    <title>YOLOv5 Demo</title>
    <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; }
        h2 { color: #333; }
        input[type=file] { margin: 20px 0; }
        img { max-width: 80%; margin-top: 20px; border: 1px solid #ddd; }
        .container { display: flex; flex-direction: column; align-items: center; }
        pre { text-align: left; background: #f0f0f0; padding: 10px; max-width: 80%; overflow-x: auto; }
    </style>
</head>
<body>
<div class="container">
<h2>YOLOv5 Image Detection</h2>
<form id="upload-form-img" method="post" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required>
    <br>
    <input type="submit" value="Upload & Detect (Image)">
</form>

<form id="upload-form-json" method="post" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required>
    <br>
    <input type="submit" value="Upload & Detect (JSON)">
</form>

<img id="result-image" src="" alt="">
<pre id="result-json"></pre>
</div>

<script>
const formImg = document.getElementById('upload-form-img');
const formJson = document.getElementById('upload-form-json');
const resultImg = document.getElementById('result-image');
const resultJson = document.getElementById('result-json');

formImg.addEventListener('submit', function(e) {
    e.preventDefault();
    const fileInput = formImg.querySelector('input[name="image"]');
    if (!fileInput.files.length) return;

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);

    fetch('/predict-img', { method: 'POST', body: formData })
        .then(resp => resp.blob())
        .then(blob => {
            resultImg.src = URL.createObjectURL(blob);
        })
        .catch(err => alert("Error: " + err));
});

formJson.addEventListener('submit', function(e) {
    e.preventDefault();
    const fileInput = formJson.querySelector('input[name="image"]');
    if (!fileInput.files.length) return;

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);

    fetch('/predict', { method: 'POST', body: formData })
        .then(resp => resp.json())
        .then(data => {
            resultJson.textContent = JSON.stringify(data, null, 2);
        })
        .catch(err => alert("Error: " + err));
});
</script>
</body>
</html>
"""

# -------------------------
# Routes
# -------------------------
@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_PAGE)

# Endpoint คืน JSON + จำนวน object
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')

    results = model(np.array(img))
    detections = results.pandas().xyxy[0]  # pandas dataframe

    output = []
    for _, row in detections.iterrows():
        output.append({
            "xmin": int(row['xmin']),
            "ymin": int(row['ymin']),
            "xmax": int(row['xmax']),
            "ymax": int(row['ymax']),
            "confidence": float(round(row['confidence'], 2)),
            "class": int(row['class']),
            "name": row['name']
        })

    return jsonify({
        "num_objects": len(output),
        "predictions": output
    })

# Endpoint คืนรูปพร้อม bounding box
@app.route('/predict-img', methods=['POST'])
def predict_img():
    if 'image' not in request.files:
        return "No image provided", 400

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')

    results = model(np.array(img))
    rendered = results.render()[0]  # numpy array
    rendered_img = Image.fromarray(rendered)

    img_io = io.BytesIO()
    rendered_img.save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


# -------------------------
# Run server
# -------------------------
if __name__ == '__main__':
    import os
    import webbrowser
    webbrowser.open(f'http://localhost:{PORT}')
    app.run(host='0.0.0.0', port=PORT)


