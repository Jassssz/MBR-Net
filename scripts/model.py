import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Dropout
from keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError
from keras.callbacks import ModelCheckpoint,EarlyStopping


def huber_fn(y_true, y_pred):
    delta = 5
    error = y_true - y_pred
    is_small_error = tf.abs(error) < delta
    squared_loss = tf.square(error) / 2
    linear_loss  = tf.multiply(tf.abs(error), delta) - 0.5 * delta**2
    return tf.where(is_small_error, squared_loss, linear_loss)

def MSE_function(y_true, y_pred):
    return K.mean(
            tf.square(tf.subtract(y_true, y_pred))
        )

def huber_model(index, 
              X1_arr_train_shuffle, 
              X2_arr_train_shuffle, 
              Y_arr_train_shuffle, 
              X1_arr_valid_shuffle, 
              X2_arr_valid_shuffle,
              Y_arr_valid_shuffle):
    num_encoder_features = 3  # Number of input features
    num_decoder_features = 6
    encoder_seq_len = 900
    decoder_seq_len = 180
    hidden_dim = 16  # Hidden dimension of LSTM
    hidden_dim2 = 16
    dropout_rate = 0.1
    learning_rate = 0.005

    # Build the encoder
    encoder_inputs = tf.keras.Input(shape=(encoder_seq_len, num_encoder_features))
    encoder_lstm1 = LSTM(hidden_dim, return_state=True, return_sequences=True)
    encoder_outputs1, state_h1, state_c1 = encoder_lstm1(encoder_inputs)
    encoder_outputs1 = Dropout(dropout_rate)(encoder_outputs1)
    encoder_lstm2 = LSTM(hidden_dim2, return_state=True, return_sequences=True)
    encoder_outputs2, state_h, state_c = encoder_lstm2(encoder_outputs1)
    encoder_states = [state_h, state_c]

    # Build the decoder
    decoder_inputs = tf.keras.Input(shape=(decoder_seq_len, num_decoder_features))
    decoder_lstm = LSTM(hidden_dim2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense_1 = Dense(hidden_dim2/2,activation='softmax')
    decoder_outputs = decoder_dense_1(decoder_outputs)
    decoder_dense_2 = Dense(1)
    decoder_outputs = decoder_dense_2(decoder_outputs)

    # Build the model
    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=huber_fn,metrics=[RootMeanSquaredError(),MeanAbsolutePercentageError(),MeanAbsoluteError()])
    model.summary()

    batch_size = 1024
    epochs = 200
    folder_name = './model/'
    file_name = 'R1616_Dropout.1_bs1024_lr0.005_Huber5_softmax' + str(index)
    cp1 = ModelCheckpoint(
        filepath=folder_name + file_name + '.h5',
        save_best_only=True, verbose=1
    )
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    # early_stopping = EarlyStopping(patience=10, restore_best_weights=True, start_from_epoch=60)
    
    # Train the model
    history=model.fit([X1_arr_train_shuffle, X2_arr_train_shuffle], Y_arr_train_shuffle, validation_data=([X1_arr_valid_shuffle, X2_arr_valid_shuffle], Y_arr_valid_shuffle),
                 epochs=epochs, batch_size=batch_size,
                 callbacks=[cp1,early_stopping])
    return history

def MSE_model(index, 
              X1_arr_train_shuffle, 
              X2_arr_train_shuffle, 
              Y_arr_train_shuffle, 
              X1_arr_valid_shuffle, 
              X2_arr_valid_shuffle,
              Y_arr_valid_shuffle):
    num_encoder_features = 3  # Number of input features
    num_decoder_features = 6
    encoder_seq_len = 900
    decoder_seq_len = 180
    hidden_dim = 16  # Hidden dimension of LSTM
    hidden_dim2 = 8
    dropout_rate = 0.1
    learning_rate = 0.005

    # Build the encoder
    encoder_inputs = tf.keras.Input(shape=(encoder_seq_len, num_encoder_features))
    encoder_lstm1 = LSTM(hidden_dim, return_state=True, return_sequences=True)
    encoder_outputs1, state_h1, state_c1 = encoder_lstm1(encoder_inputs)
    encoder_outputs1 = Dropout(dropout_rate)(encoder_outputs1)
    encoder_lstm2 = LSTM(hidden_dim2, return_state=True, return_sequences=True)
    encoder_outputs2, state_h, state_c = encoder_lstm2(encoder_outputs1)
    encoder_states = [state_h, state_c]

    # Build the decoder
    decoder_inputs = tf.keras.Input(shape=(decoder_seq_len, num_decoder_features))
    decoder_lstm = LSTM(hidden_dim2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense_1 = Dense(hidden_dim2/2,activation='relu')
    decoder_outputs = decoder_dense_1(decoder_outputs)
    decoder_dense_2 = Dense(1)
    decoder_outputs = decoder_dense_2(decoder_outputs)

    # Build the model
    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MSE_function,metrics=[RootMeanSquaredError(),MeanAbsolutePercentageError(),MeanAbsoluteError()])
    model.summary()

    batch_size = 1024
    epochs = 200
    folder_name = './model/'
    file_name = 'R168_Dropout.1_bs1024_lr0.005_MSE' + str(index)
    cp1 = ModelCheckpoint(
        filepath=folder_name + file_name + '.h5',
        save_best_only=True, verbose=1
    )
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    # early_stopping = EarlyStopping(patience=10, restore_best_weights=True, start_from_epoch=60)
    
    # Train the model
    history=model.fit([X1_arr_train_shuffle, X2_arr_train_shuffle], Y_arr_train_shuffle, validation_data=([X1_arr_valid_shuffle, X2_arr_valid_shuffle], Y_arr_valid_shuffle),
                 epochs=epochs, batch_size=batch_size,
                 callbacks=[cp1,early_stopping])
    return history
