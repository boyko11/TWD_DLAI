from td_utils import *
from model_service import ModelService
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np

if __name__ == '__main__':

    Tx = 5511  # The number of time steps input to the model from the spectrogram
    n_freq = 101  # Number of frequencies input to the model at each time step of the spectrogram

    Ty = 1375  # The number of time steps in the output of our model

    activates, negatives, backgrounds = load_raw_audio()

    model_service = ModelService()
    model = model_service.model(input_shape=(Tx, n_freq))
    model.summary()

    model = load_model('./models/tr_model.h5')

    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    X = np.load("./XY_train/X.npy")
    Y = np.load("./XY_train/Y.npy")

    X_dev = np.load("./XY_dev/X_dev.npy")
    Y_dev = np.load("./XY_dev/Y_dev.npy")

    model.fit(X, Y, batch_size=5, epochs=1)

    loss, acc = model.evaluate(X_dev, Y_dev)
    print("Dev set accuracy = ", acc)