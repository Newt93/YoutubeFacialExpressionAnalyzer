import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

class FacialExpressionClassifier:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.class_names = None
        self.num_classes = None

    def load_data(self, data_path):
        # load image data and labels here
        pass

    def create_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

    def train(self):
        self.model.fit(self.X_train, self.y_train, batch_size=32, epochs=10,
                       validation_data=(self.X_test, self.y_test))

    def evaluate(self):
        score = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def predict(self, image):
        image = image.reshape(1, 48, 48, 1)
        predictions = self.model.predict(image)
        return self.class_names[np.argmax(predictions)]

    def save(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

if __name__ == "__main__":
    classifier = FacialExpressionClassifier()
    classifier.load_data("path/to/data")
    classifier.create_model()
    classifier.train()
    classifier.evaluate()
    classifier.save("expressions_classifier.pkl")

