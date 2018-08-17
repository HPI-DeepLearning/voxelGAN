import pylab as pl
import matplotlib.cm as cm
import numpy as np

from keras import backend as K
from model import *
from utils import *
from config import *

from keras.optimizers import Adam

opt = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
  
gen = Generator((sequence_length, size, size, input), output, kernel_depth, size*size*sequence_length)
gen.compile(loss='mae', optimizer=opt)
gen.load_weights(checkpoint_gen_name)

model = load_model(checkpoint_gen_name)
model.summary()

convout32_f = model.get_layer(conv_32).output
#convout16_f= model.get_layer(conv_16).output
#convout8_f= model.get_layer(conv_8).output
#upconvout16_f= model.get_layer(up_conv_16).output
#upconvout32_f= model.get_layer(up_conv_32).output


# List sequences  
sequences = prepare_data(test_dir)

progbar = keras.utils.Progbar(len(sequences))

for s in range(len(sequences)):
    
    progbar.add(1)
    sequence = sequences[s]
    x, y = load(sequence, sequence_length)
    
    for i in range(len(x)):
    
        # predict
        C1=convout32_f(x)
        C1= np.squeeze(C1)
        print("C1 shape : ",C1.shape)
        Pl.figure(figsize=(15,15))
        pl.suptitle('conv1out')
        nice_imshow(pl.gca(), make_mosaic(C1, 6, 6), cmap=cm.binary)


        save_image(x[i] / 2 + 0.5, y[i], re_shape(generated_y), convvis_dir + "conv32{}.png".format(s)) ???
