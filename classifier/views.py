from django.shortcuts import render

import librosa
import numpy as np
from django.shortcuts import render
from .forms import AudioUploadForm
# from .ml_model import predict_audio_class

def extract_features(file):
    y, sr = librosa.load(file, duration=5.0)  # 5 seconds of audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def upload_audio(request):
    if request.method == 'POST' and request.FILES['audio_file']:
        audio_file = request.FILES['audio_file']
        # features = extract_features(audio_file)
        # result = predict_audio_class(features)
        result = "Normal"
        
        return render(request, 'classifier/result.html', {'result': result})
    return render(request, 'classifier/upload.html')

