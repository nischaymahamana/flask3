from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import librosa
import io, soundfile
import requests

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

def download_model(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

model_url = "https://mega.nz/file/d9FWgKID#trQEPKCfK0gMAz07IFmD42h9XJDX5a2lk7IeTfB9pII"
download_model(model_url, 'speech_model.pkl')
model=""
joblib.dump(model, 'speech_model.pkl', compress=True)


def extract_feature(file_data, mfcc, chroma, mel):
    try:
        with soundfile.SoundFile(io.BytesIO(file_data)) as sound_file:
            X = sound_file.read(dtype="float32")
            if X.ndim == 2:
                X = librosa.to_mono(X)
            sample_rate = sound_file.samplerate

            if chroma:
                stft = np.abs(librosa.stft(X))

            result = np.array([])

            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result = np.hstack((result, mfccs))

            if chroma:
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma))

            if mel:
                mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
                result = np.hstack((result, mel))

            return result

    except Exception as e:
        print(f"Error in extract_feature: {str(e)}")
        return np.array([])

@app.route('/', methods=['GET'])
def home():
    return "<h1> API </h1>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'voice' not in request.files:
            return jsonify({'error': 'No voice file provided'})

        voice_data = request.files['voice'].read()

        features = extract_feature(voice_data, mfcc=True, chroma=True, mel=True)

        print("Extracted Features:", features)

        if not features.any():
            return jsonify({'error': 'No features extracted from the audio'}), 500

        predictions = model.predict(features.reshape(-1, 1))

        return jsonify({'predictions': predictions.tolist() if predictions else 'No predictions available'})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
