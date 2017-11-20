from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.layers.recurrent import LSTM
import keras.backend as K
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn import preprocessing

def build_autoencoder():
    input_size = 2048
    hidden_size = 512
    code_size = 128

    input_img = Input(shape=(input_size,))
    hidden_1 = BatchNormalization()(input_img)
    hidden_1 = Dense(hidden_size, activation='relu')(input_img)
    code = Dense(code_size, activation='relu')(hidden_1)
    hidden_2 = Dense(hidden_size, activation='relu')(code)
    output_img = Dense(
        input_size, activation='relu')(hidden_2)

    autoencoder = Model(input_img, output_img)
    num_layers = len(autoencoder.layers) - 2
    return autoencoder

def main():
    model = build_autoencoder()
    nb_epoch = 100
    batch_size = 100

    train_data = np.loadtxt('../data/features/impacts_period1.npy')
    # test_data = np.loadtxt('../data/features/period1_full.npy')

    min_max_scale = preprocessing.MinMaxScaler();
    train_data = min_max_scale.fit_transform(train_data)
    model.compile(optimizer='adadelta', loss='mean_squared_error')

    checkpointer = ModelCheckpoint(filepath="./checkpoints/autoencoder_model.h5",
                                   verbose=1,
                                   save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)
    history = model.fit(x=train_data + (np.random.rand(train_data.shape[0], train_data.shape[1])), y=train_data,
                        epochs=nb_epoch, batch_size=batch_size, shuffle=True, verbose=1, callbacks=[checkpointer, tensorboard], validation_split=0.2).history

    model.save_weights('../data/weights/autoencoder_weights.h5')

if __name__ == '__main__':
    main()
