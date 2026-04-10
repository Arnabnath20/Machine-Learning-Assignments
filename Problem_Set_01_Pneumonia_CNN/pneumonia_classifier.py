import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_dir = 'chest_xray/train'
val_dir = 'chest_xray/val'
test_dir = 'chest_xray/test'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


test_datagen = ImageDataGenerator(rescale=1./255)

print("Loading dataset...")

train_set = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(150, 150), 
    batch_size=32, 
    class_mode='binary'
)

val_set = test_datagen.flow_from_directory(
    val_dir, 
    target_size=(150, 150), 
    batch_size=16, 
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    test_dir, 
    target_size=(150, 150), 
    batch_size=32, 
    class_mode='binary'
)


model = Sequential()


model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2, 2))


model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))


model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))


model.add(Dropout(0.4)) 

model.add(Dense(1, activation='sigmoid')) 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.summary()

print("Starting to train the model...")
history = model.fit(
    train_set,
    epochs=10,
    validation_data=val_set
)

print("Testing model accuracy...")
loss, accuracy = model.evaluate(test_set)
print(f"Accuracy on test data: {accuracy * 100:.2f}%")

model.save('my_pneumonia_model.h5')
print("Model saved successfully!")
