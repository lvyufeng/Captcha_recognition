
def simple_cnn():
    model = Sequential()
    model.add(Convolution2D(32, 1, 4, 4, border_mode='full', activation='relu'))
    model.add(Convolution2D(32, 32, 4, 4, activation='relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, 32, 4, 4, border_mode='full', activation='relu'))
    model.add(Convolution2D(64, 64, 4, 4, activation='relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64 * 5 * 5, 512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, nb_class, activation='softmax'))
    # model.load_weights(model_path)
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')
    return model