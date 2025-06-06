from tensorflow.keras.models import load_model
import numpy as np

model = load_model("enhanced_model_binary.h5")

def predict_audio_class(features):
    features = np.reshape(features, (1, -1))  # Make it (1, 36)
    prediction = model.predict(features)
    return prediction