from tensorflow.keras.models import load_model
import numpy as np

multi_model = load_model("enhanced_model_4k.h5")

def predict_audio_class_multi(features):
    features = np.reshape(features, (1, -1))  # Make it (1, 36)
    prediction = multi_model.predict(features)
    return prediction