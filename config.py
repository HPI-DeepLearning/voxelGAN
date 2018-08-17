import os
from keras import backend as K

K.set_image_data_format('channels_last')

# general
data_dir = 'data/train/**/*t1.nii.gz'
predict_dir = 'data/predict/**/*t1.nii.gz'
checkpoint_dir = 'checkpoints/'

directory = os.path.dirname(checkpoint_dir)
if not os.path.exists(directory):
    os.makedirs(directory)

# bdsscgan
input = 4
output = 4
size = 32
epochs = 50
kernel_depth = 32

checkpoint_gen_name = checkpoint_dir + 'gen.hdf5'
checkpoint_disc_name = checkpoint_dir + 'disc.hdf5'
