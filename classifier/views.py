from django.shortcuts import render

import librosa
import numpy as np
import pywt
from django.shortcuts import render
from .forms import AudioUploadForm
from .binary_model import predict_audio_class
from .multi_class_model import predict_audio_class_multi
from azure.storage.blob import BlobServiceClient
from django.conf import settings


# Function to extract MFCC features.
def extract_mfcc_features(signal, sr, n_mfcc=16):
    # Extract MFCCs over the entire signal.
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    # Compute summary statistics: mean and standard deviation per coefficient.
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    # Concatenate mean and standard deviation features.
    return np.concatenate([mfcc_mean, mfcc_std])  # Resulting in 2*n_mfcc features

# Function to extract DWT features.
def extract_dwt_features(signal, wavelet='db4', level=3):
    # Compute the discrete wavelet transform.
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # For each decomposition level, compute the energy of the coefficients.
    energies = [np.sum(np.square(c)) for c in coeffs]
    return np.array(energies)  # This will return (level+1) features.

def extract_features(file):
    signal, sr = librosa.load(file, duration=5.0)  # 5 seconds of audio
     # Extract MFCC features.
    mfcc_feats = extract_mfcc_features(signal, sr, n_mfcc=16)
    # Extract DWT features.
    dwt_feats = extract_dwt_features(signal, wavelet='db4', level=3)
    # Combine MFCC and DWT features.
    combined_features = np.concatenate([mfcc_feats, dwt_feats])
    return combined_features

def upload_audio(request):
    if request.method == 'POST' and request.FILES['audio_file']:
        audio_file = request.FILES['audio_file']

        # ✅ Upload to Azure Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(settings.AZURE_CONTAINER_NAME)
        blob_client = container_client.get_blob_client(audio_file.name)
        blob_client.upload_blob(audio_file, overwrite=True)
        audio_file.seek(0)
        
        features = extract_features(audio_file)
        result = predict_audio_class(features)
        index = np.argmax(result)
        if index == 0:
            label = "Normal"
        else:
            label = "Abnormal"
        result = label
        if label == "Abnormal":
            result2 = predict_audio_class_multi(features)
            class_labels = ["Aortic Stenosis", "Mitral Regurgigation", "Miteral Stenosis", "MVP", "Normal"]
            # Get index of highest probability
            predicted_index = np.argmax(result)
            # Map to label
            predicted_label = class_labels[predicted_index]
            result = label + " ,  " + predicted_label
        
        return render(request, 'classifier/result.html', {'result': result})
    return render(request, 'classifier/upload.html')

