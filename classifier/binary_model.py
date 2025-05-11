import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import librosa
import pywt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ----- CONFIGURATION -----
csv_path = r'D:\PycharmProjects\FYP\Classification-of-Heart-Sound-Signal-Using-Multiple-Features--master/Classification-of-Heart-Sound-Signal-Using-Multiple-Features--master/heart_sound_features.csv'  # Change to your desired directory
model_save_path = r"D:\PycharmProjects\FYP/Classification-of-Heart-Sound-Signal-Using-Multiple-Features--master/Classification-of-Heart-Sound-Signal-Using-Multiple-Features--master/enhanced_model_.h5"  # Path to save the .h5 model
RANDOM_STATE = 42

# ----- LOAD DATA -----
df = pd.read_csv(csv_path)
print("Loaded data shape:", df.shape)

# Encode label if necessary
if df['label'].dtype == 'object':
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

# Split features and labels
X = df.drop('label', axis=1).values
y = df['label'].values
y_cat = to_categorical(y, num_classes=2)

# ----- SPLIT DATA -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cat
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# ----- BUILD THE MODEL -----
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----- CALLBACKS -----
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

# ----- TRAIN THE MODEL -----
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ----- EVALUATE THE MODEL -----
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# ----- CONFUSION MATRIX & REPORT -----
cm = confusion_matrix(y_true, y_pred)
print("Classification Report:")
print(classification_report(y_true, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[str(i) for i in range(2)],
            yticklabels=[str(i) for i in range(2)])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# ----- SAVE MODEL AS .H5 -----
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"Model saved successfully at: {model_save_path}")