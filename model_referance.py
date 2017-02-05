
net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=1000,
    verbose=1,
    )

X, y = load2d()  # load 2-d data
net2.fit(X, y)

# Training for 1000 epochs will take a while.  We'll pickle the
# trained model so that we can load it back later:
import cPickle as pickle
with open('net2.pickle', 'wb') as f:
    pickle.dump(net2, f, -1)
"""
InputLayer            (None, 1, 96, 96)       produces    9216 outputs
Conv2DCCLayer         (None, 32, 94, 94)      produces  282752 outputs
MaxPool2DCCLayer      (None, 32, 47, 47)      produces   70688 outputs
Conv2DCCLayer         (None, 64, 46, 46)      produces  135424 outputs
MaxPool2DCCLayer      (None, 64, 23, 23)      produces   33856 outputs
Conv2DCCLayer         (None, 128, 22, 22)     produces   61952 outputs
MaxPool2DCCLayer      (None, 128, 11, 11)     produces   15488 outputs
DenseLayer            (None, 500)             produces     500 outputs
DenseLayer            (None, 500)             produces     500 outputs
DenseLayer            (None, 30)              produces      30 outputs
"""