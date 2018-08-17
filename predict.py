from keras.optimizers import Adadelta

from model import *
from utils import *
from config import *

opt = Adadelta()
  
gen = Generator((size, size, size, input), output, kernel_depth)
gen.compile(loss='mae', optimizer=opt)
gen.load_weights(checkpoint_gen_name)

# List sequences  
sequences = prepare_data(predict_dir)
print(sequences)

progbar = keras.utils.Progbar(len(sequences))

for s in range(len(sequences)):
    
    
    progbar.add(1)
    sequence = sequences[s]
    x, idx = load2(sequence, size)
    y = []
    
    for i in range(len(x)):
    
        # gen
        fake = gen.predict(x[i])
        print(fake.shape)
        
        y.append(fake)
        
    store(sequence, y, idx)
    
    #save_image(x[i] / 2 + 0.5, y[i], re_shape(generated_y), prediction_dir + "test{}.png".format(s))
