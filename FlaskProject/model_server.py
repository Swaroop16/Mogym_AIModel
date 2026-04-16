import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import pose_model  # your existing AI model module
app = Flask(__name__)
CORS(app)  # allow requests from your HTML frontend

@app.route('/predict', methods=['POST'])
def predict():
    video_file = request.files['video']
    exercise = request.form.get('exercise')
    # Pass video to your model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name
    result = pose_model.analyze(tmp_path,exercise)
    os.remove(tmp_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5001)