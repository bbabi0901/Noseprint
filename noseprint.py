import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.models import Model

from resnet import ResNet
from hourglass import Hourglass



class Train():
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 224
    PATIENTCE = 20

    def __init__(self, dataset, architecture):
        if not str(dataset).endswith('.npy'):
            raise ImportError("Dataset should be npy file")

        assert architecture in ['resnet50', 'resnet101', 'hourglass']
        self.ARCHITECTURE = architecture

        self.train_data, self.val_data = self._load_dataset(dataset) 
        self.noseprint_model = self.set_model(self.ARCHITECTURE, self.num_landmarks)

    def set_model(self, architecture, num_landmarks):
        inputs = Input(shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3))
        if architecture == "hourglass":
            x = Hourglass().hourglass_model(inputs=inputs, num_landmarks=num_landmarks, num_channels=self.IMAGE_SIZE)
        else:
            x = ResNet().resnet_model(inputs=inputs, architecture=architecture)
        x = Dropout(0.4)(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        outputs = Dense(num_landmarks, activation='linear')(x)

        return Model(inputs=inputs, outputs=outputs)

    def _load_dataset(self, path, test_size=0.3, shuffle=True):
        data = np.load(path, allow_pickle=True)

        x_data = np.asarray(data.item().get('img')).astype(np.float32)
        y_data = np.asarray(data.item().get('landmark')).astype(np.float32)

        if x_data.shape[1:] != (self.IMAGE_SIZE, self.IMAGE_SIZE, 3):
            raise ValueError('shape of input image must be (224, 224, 3)')

        self.num_landmarks = y_data.shape[1]

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, shuffle=shuffle)

        x_train = np.reshape(x_train / 255, (-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 3))
        x_test = np.reshape(x_test / 255, (-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 3))
        y_train = np.reshape(y_train, (-1, self.num_landmarks))
        y_test = np.reshape(y_test, (-1, self.num_landmarks))

        return (x_train, y_train), (x_test, y_test)

    def train(self, epochs, batch_size, model_dir):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        self.noseprint_model.compile(optimizer='adam', loss='mse', metrics=['acc'])

        callbacks = [
            EarlyStopping(patience=self.PATIENTCE),
            ModelCheckpoint(filepath=model_dir + '/noseprint_{epoch:02d}-{val_loss:.2f}.h5',
                            moniter='val_loss',
                            save_best_only=True,
                            save_weights_only=True
                            ),
            LearningRateScheduler(self._scheduler)
        ]

        self.noseprint_model.fit(self.train_data[0], self.train_data[1], epochs=epochs, batch_size=batch_size, shuffle=True,
                  validation_data=self.val_data, verbose=1, callbacks=callbacks)

    def _scheduler(self, epoch):
        if epoch > 0 and epoch % 50 == 0:
            self.LEARNING_RATE = self.LEARNING_RATE / 10
        return self.LEARNING_RATE

class Inference(Train):

    def __init__(self, weights, architecture, num_landmarks):
        if not str(weights).endswith('.h5'):
            raise ImportError("Weights should be h5py file.")

        assert architecture in ['resnet50', 'resnet101', 'hourglass']
        self.ARCHITECTURE = architecture
        self.num_landmarks = num_landmarks * 2

        self.noseprint_model = self.set_model(self.ARCHITECTURE, self.num_landmarks)
        self.noseprint_model.load_weights(weights)
        print("Weights loaded.")

    def get_landmarks(self, image):
        self.img = cv2.imread(image)
        self.INPUT_SHAPE = self.img.shape
        if self.INPUT_SHAPE != (self.IMAGE_SIZE, self.IMAGE_SIZE, 3):
            img_rsz, ratio, top, left = self._resize_img()
            inputs = (img_rsz.astype('float32') / 255.).reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 3)
            landmarks = self.noseprint_model.predict(inputs)[0].reshape(-1, 2)
            landmarks = (landmarks - np.array([left, top])) / ratio

        else:
            inputs = (self.img.astype('float32') / 255.).reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 3)
            landmarks = self.noseprint_model.predict(inputs)[0].reshape(-1, 2)

        return landmarks

    def _resize_img(self):
        input_size = self.INPUT_SHAPE[:2]
        ratio = float(self.IMAGE_SIZE) / max(input_size)
        new_size = tuple([int(x * ratio) for x in input_size])

        img_rsz = cv2.resize(self.img, (new_size[1], new_size[0]))
        delta_w = self.IMAGE_SIZE - new_size[1]
        delta_h = self.IMAGE_SIZE - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        new_im = cv2.copyMakeBorder(img_rsz, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return new_im, ratio, top, left