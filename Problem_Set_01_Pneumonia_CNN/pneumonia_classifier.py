import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = 'chest_xray/train'
val_path = 'chest_xray/val'
test_path = 'chest_xray/test'

print("Initializing Data Preprocessing...")


train_generator_setup = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


test_val_generator_setup = ImageDataGenerator(rescale=1.0/255.0)


train_data = train_generator_setup.flow_from_directory(
    train_path, target_size=(150, 150), batch_size=32, class_mode='binary'
)

val_data = test_val_generator_setup.flow_from_directory(
    val_path, target_size=(150, 150), batch_size=16, class_mode='binary'
)

test_data = test_val_generator_setup.flow_from_directory(
    test_path, target_size=(150, 150), batch_size=32, class_mode='binary'
)


print("Building the CNN Model...")
cnn_model = Sequential()


cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))


cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))


cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))


cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.4)) # Adding dropout to reduce overfitting
cnn_model.add(Dense(1, activation='sigmoid')) # Binary output (Normal vs Pneumonia)


cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


print("Starting Model Training...")
model_history = cnn_model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)


print("Evaluating Model on Test Data...")
test_loss, test_accuracy = cnn_model.evaluate(test_data)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")


cnn_model.save('pediatric_pneumonia_cnn.h5')
print("Model saved successfully as pediatric_pneumonia_cnn.h5")
