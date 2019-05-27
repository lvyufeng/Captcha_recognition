import numpy as np
from keras import backend as K
from keras.layers import Input,Conv2D,Lambda,merge,Dense,Flatten,MaxPooling2D
from keras.models import Model,Sequential
from keras.regularizers import l2
from keras.losses import binary_crossentropy
from keras.optimizers import Adam


def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = np.random.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=np.random.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

def siamese_base(input_shape):
    convnet = Sequential()
    convnet.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                       kernel_initializer=W_init, kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128, (3, 3), activation='relu',
                       kernel_regularizer=l2(2e-4), kernel_initializer=W_init, bias_initializer=b_init))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=W_init, kernel_regularizer=l2(2e-4),
                       bias_initializer=b_init))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(256, (2, 2), activation='relu', kernel_initializer=W_init, kernel_regularizer=l2(2e-4),
                       bias_initializer=b_init))
    convnet.add(Flatten())
    convnet.add(Dense(4096, activation="sigmoid", kernel_regularizer=l2(1e-3), kernel_initializer=W_init,
                      bias_initializer=b_init))
    return convnet

def build_model():
    input_shape_l = (32, 128, 1)
    input_shape_r = (32, 175, 1)
    left_input = Input(input_shape_l)
    right_input = Input(input_shape_r)
    #build convnet to use in each siamese 'leg'
    convnet_l = siamese_base(input_shape_l)
    convnet_r = siamese_base(input_shape_r)
    #encode each of the two inputs into a vector with the convnet
    encoded_l = convnet_l(left_input)
    encoded_r = convnet_r(right_input)
    #merge two encoded inputs with the l1 distance between them
    L1_distance = lambda x: K.abs(x[0]-x[1])
    both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
    prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(both)
    siamese_net = Model(input=[left_input,right_input],output=prediction)
    #optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)

    optimizer = Adam(0.00006)
    #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])
    return siamese_net
