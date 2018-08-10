"""
 * Created by PyCharm.
 * User: tuhoangbk
 * Date: 10/08/2018
 * Time: 11:50
 * Have a nice day　:*)　:*)
"""

import keras
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Activation
from keras.applications.densenet import DenseNet121

def build_densenet():
    densenet = DenseNet121(include_top=True,
                       weights=None,
                       input_shape=(480, 480, 3),
                       classes=103)
    model = Sequential([densenet, Activation('softmax')])

    return densenet
    # return model


def fit_model():
    pass

def test_model():
    pass

if __name__ == '__main__':
    model = build_densenet()
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])