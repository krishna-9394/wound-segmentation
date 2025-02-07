import os
from models.deeplab import Deeplabv3, relu6, BilinearUpsampling
from tensorflow.keras.layers import DepthwiseConv2D
from utils.learning.metrics import dice_coef, precision, recall
from utils.io.data import load_test_images, save_results, DataGen

# Settings
input_dim_x = 224
input_dim_y = 224
path = './data/chronic_wound_ulcer/'
file_path = './training_history/2019-12-19 01%3A53%3A15.480800.hdf5'
pred_save_path = './predictions/'

# Rebuild the model architecture
model = Deeplabv3(input_shape=(input_dim_x, input_dim_y, 3), classes=1)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, precision, recall])

# Load weights into the model
model.load_weights(file_path)

# Prepare the data
data_gen = DataGen(path, split_ratio=0.0, x=input_dim_x, y=input_dim_y, color_space='rgb')

# Predict and save results
test_image_files = os.listdir(os.path.join(path, "test/images/"))
for image_batch, _ in data_gen.generate_data(batch_size=len(test_image_files), test=True):
    prediction = model.predict(image_batch, verbose=1)
    save_results(prediction, 'rgb', pred_save_path, test_image_files)
    break
