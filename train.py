from keras.callbacks import TensorBoard
from keras.optimizers import Adadelta

from model import *
from utils import *
from config import *

# Create optimizers
opt_dcgan = Adadelta()
opt_discriminator = Adadelta()
  
gen = Generator((size, size, size, input), output, kernel_depth)
gen.compile(loss='mae', optimizer=opt_discriminator)

disc = Discriminator((size, size, size, input), (size, size, size, output), kernel_depth)
disc.trainable = False

combined = Combine(gen, disc, (size, size, size, input), (size, size, size, output))
loss = [selective_crossentropy, 'binary_crossentropy']
loss_weights = [10, 1]
combined.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

disc.trainable = True
disc.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

if os.path.isfile(checkpoint_gen_name):
    gen.load_weights(checkpoint_gen_name)
if os.path.isfile(checkpoint_disc_name):
    disc.load_weights(checkpoint_disc_name)

# List sequences  
sequences = prepare_data(data_dir)
print(sequences)

real_y = np.reshape(np.array([0, 1]), (1, 2))
fake_y = np.reshape(np.array([1, 0]), (1, 2))

#log = open("train.log",'w')

tensorlog = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=1, write_graph=True, write_grads=True, write_images=True)
tensorlog.set_model(gen)

for e in range(epochs):
    print("Epoch {}".format(e))
    random.shuffle(sequences)
    
    # select a fraction
    train_offset = int(len(sequences) * 0.9)
    train_sequence = sequences[:train_offset]
    
    progbar = keras.utils.Progbar(len(train_sequence))
    
    for s in range(len(train_sequence)):
        
        progbar.add(1)
        sequence = train_sequence[s]
        x, y, idx = load(sequence, size)
        
        for i in range(len(x)):
        
            # train disc on real
            disc.train_on_batch([x[i], y[i]], real_y)
        
            # gen fake
            fake = gen.predict(x[i])
        
            # train disc on fake
            disc.train_on_batch([x[i], fake], fake_y)
        
            # train combined    
            disc.trainable = False
            combined.train_on_batch(x[i], [y[i], real_y])
            disc.trainable = True
            
            #log.write(str(e) + ", " + str(s) + ", " + str(dr_loss) + ", " + str(df_loss) + ", " + str(g_loss[0]) + ", " + str(g_loss[1]) + ", " + str(opt_dcgan.get_config()["lr"]) + "\n")
            
    # output random result
    #val_sequence = sequences[train_offset:]
    #generated_y = gen.predict(x[random_index])
    #save_image(strip(x[random_index]) / 2 + 0.5, y[random_index], re_shape(generated_y), "validation/e{}_{}.png".format(e, s))
        
    # save weights
    gen.save_weights(checkpoint_gen_name, overwrite=True)
    disc.save_weights(checkpoint_disc_name, overwrite=True)
    
    tensorlog.on_epoch_end(e)
    
tensorlog.on_train_end()
