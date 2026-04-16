from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import tempfile, os

app = Flask(__name__)
CORS(app)

# Forward video to AI model server
@app.route('/analyze', methods=['POST'])
def analyze():
    video = request.files['video']
    exercise = request.form.get('exercise')
    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video.save(tmp.name)
        tmp_path = tmp.name

    # Send to AI model server running on port 5001
    with open(tmp_path, 'rb') as f:
        response = requests.post(
            'http://localhost:5001/predict',
            files={'video': f},
            data={'exercise': exercise}
        )

    # Clean up temp file
    os.remove(tmp_path)

    return jsonify(response.json())
    if response.status_code != 200:
        return jsonify({"error": "Model server failed"}), 500

if __name__ == '__main__':
    app.run(port=5000)
