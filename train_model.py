import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Paths
train_dir = r"D:\emotion detection\data\train"
test_dir = r"D:\emotion detection\data\test"

# Image size and batch size
img_size = (48, 48)
batch_size = 32

# Load train and test datasets
# Load train and test datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale"
)

# ✅ Save class names before mapping
class_names = train_ds.class_names
print("Detected classes:", class_names)

# Normalize (0–1 range)
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Build CNN model
model = Sequential([
    tf.keras.layers.Input(shape=(48, 48, 1)),   # ✅ Use Input layer instead of input_shape in Conv2D
    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(class_names), activation="softmax")  # ✅ use stored class names
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(train_ds, validation_data=test_ds, epochs=20)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/emotion_model.h5")
print("✅ Model trained and saved as model/emotion_model.h5")
