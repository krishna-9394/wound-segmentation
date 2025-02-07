import os
import cv2
import json
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt


class DataGen:

    def __init__(self, path, split_ratio, x, y, color_space='rgb'):
        self.x = x
        self.y = y
        self.path = path
        self.color_space = color_space

        # Paths
        self.path_train_images = os.path.join(path, "train/images/")
        self.path_train_labels = os.path.join(path, "train/labels/")
        self.path_test_images = os.path.join(path, "test/images/")
        self.path_test_labels = os.path.join(path, "test/labels/")

        # Test data
        self.x_test_file_list = get_png_filename_list(self.path_test_images)
        self.y_test_file_list = get_png_filename_list(self.path_test_labels)

        # Handle missing training data
        self.image_file_list = get_png_filename_list(self.path_train_images)
        self.label_file_list = get_png_filename_list(self.path_train_labels)

        if self.image_file_list and self.label_file_list:
            self.image_file_list[:], self.label_file_list[:] = self.shuffle_image_label_lists_together()
            self.split_index = int(split_ratio * len(self.image_file_list))
            self.x_train_file_list = self.image_file_list[self.split_index:]
            self.y_train_file_list = self.label_file_list[self.split_index:]
            self.x_val_file_list = self.image_file_list[:self.split_index]
            self.y_val_file_list = self.label_file_list[:self.split_index]
        else:
            print("No training data found. Proceeding with test-only mode.")

    def generate_data(self, batch_size, train=False, val=False, test=False):
        """Replaces Keras' native ImageDataGenerator."""
        if test:
            image_file_list = self.x_test_file_list
            label_file_list = self.y_test_file_list
        elif train:
            image_file_list = self.x_train_file_list
            label_file_list = self.y_train_file_list
        elif val:
            image_file_list = self.x_val_file_list
            label_file_list = self.y_val_file_list
        else:
            raise ValueError("One of train, val, or test must be True.")

        i = 0
        while True:
            image_batch = []
            label_batch = []
            for b in range(batch_size):
                if i == len(image_file_list):
                    i = 0
                if i < len(image_file_list):
                    sample_image_filename = image_file_list[i]
                    if test:
                        image = cv2.imread(os.path.join(self.path_test_images, sample_image_filename), 1)
                        # Resize image to match model input shape
                        image = cv2.resize(image, (self.x, self.y))
                        label = np.zeros((self.x, self.y, 1))  # Test labels may not exist
                    else:
                        label = cv2.imread(os.path.join(self.path_train_labels, sample_image_filename), 0)
                        image = cv2.imread(os.path.join(self.path_train_images, sample_image_filename), 1)
                        image = cv2.resize(image, (self.x, self.y))
                        label = cv2.resize(label, (self.x, self.y))
                        label = np.expand_dims(label, axis=2)

                    image_batch.append(image.astype("float32"))
                    label_batch.append(label.astype("float32"))
                i += 1

            if image_batch:
                yield normalize(np.array(image_batch)), normalize(np.array(label_batch))
                
    def get_num_data_points(self, train=False, val=False):
        try:
            image_file_list = self.x_train_file_list if val is False and train is True else self.x_val_file_list
        except ValueError:
            print('one of train or val need to be True')

        return len(image_file_list)

    def shuffle_image_label_lists_together(self):
        combined = list(zip(self.image_file_list, self.label_file_list))
        random.shuffle(combined)
        return zip(*combined)

    @staticmethod
    def change_color_space(image, label, color_space):
        if color_space.lower() == 'hsi' or color_space.lower() == 'hsv':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2HSV)
        elif color_space.lower() == 'lab':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2LAB)
        return image, label
def normalize(arr):
    diff = np.amax(arr) - np.amin(arr)
    diff = 255 if diff == 0 else diff
    arr = arr / np.absolute(diff)
    return arr


def get_png_filename_list(path):
    file_list = []
    for FileNameLength in range(0, 500):
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                # check file extension
                if ".png" in filename.lower() and len(filename) == FileNameLength:
                    file_list.append(filename)
            break
    file_list.sort()
    return file_list


def get_jpg_filename_list(path):
    file_list = []
    for FileNameLength in range(0, 500):
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                # check file extension
                if ".jpg" in filename.lower() and len(filename) == FileNameLength:
                    file_list.append(filename)
            break
    file_list.sort()
    return file_list


def load_jpg_images(path):
    file_list = get_jpg_filename_list(path)
    temp_list = []
    for filename in file_list:
        img = cv2.imread(path + filename, 1)
        temp_list.append(img.astype("float32"))

    temp_list = np.array(temp_list)
    # x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    return temp_list, file_list


def load_png_images(path):

    temp_list = []
    file_list = get_png_filename_list(path)
    for filename in file_list:
        img = cv2.imread(path + filename, 1)
        temp_list.append(img.astype("float32"))

    temp_list = np.array(temp_list)
    #temp_list = np.reshape(temp_list,(temp_list.shape[0], temp_list.shape[1], temp_list.shape[2], 3))
    return temp_list, file_list


def load_data(path):
    # path_train_images = path + "train/images/padded/"
    # path_train_labels = path + "train/labels/padded/"
    # path_test_images = path + "test/images/padded/"
    # path_test_labels = path + "test/labels/padded/"
    path_train_images = path + "train/images/"
    path_train_labels = path + "train/labels/"
    path_test_images = path + "test/images/"
    path_test_labels = path + "test/labels/"
    x_train, train_image_filenames_list = load_png_images(path_train_images)
    y_train, train_label_filenames_list = load_png_images(path_train_labels)
    x_test, test_image_filenames_list = load_png_images(path_test_images)
    y_test, test_label_filenames_list = load_png_images(path_test_labels)
    x_train = normalize(x_train)
    y_train = normalize(y_train)
    x_test = normalize(x_test)
    y_test = normalize(y_test)
    return x_train, y_train, x_test, y_test, test_label_filenames_list


def load_test_images(path):
    path_test_images = path + "test/images/"
    x_test, test_image_filenames_list = load_png_images(path_test_images)
    x_test = normalize(x_test)
    return x_test, test_image_filenames_list


def save_results(np_array, color_space, outpath, test_label_filenames_list):
    # Ensure the output directory exists
    os.makedirs(outpath, exist_ok=True)
    
    # Save predictions
    for i, filename in enumerate(test_label_filenames_list):
        if i < len(np_array):  # Ensure the index is valid
            pred = np_array[i]
            if color_space.lower() == 'rgb':
                cv2.imwrite(os.path.join(outpath, filename), pred * 255.)
            else:
                print(f"Unsupported color space: {color_space}. Prediction not saved for {filename}.")
        else:
            print(f"Warning: No prediction available for {filename}.")



def save_rgb_results(np_array, outpath, test_label_filenames_list):
    i = 0
    for filename in test_label_filenames_list:
        # predict_img = np.reshape(predict_img,(predict_img[0],predict_img[1]))
        cv2.imwrite(outpath + filename, np_array[i] * 255.)
        i += 1


def save_history(model, model_name, training_history, dataset, n_filters, epoch, learning_rate, loss,
                 color_space, path=None, temp_name=None):
    save_weight_filename = temp_name if temp_name else str(datetime.datetime.now())
    model.save('{}{}.hdf5'.format(path, save_weight_filename))
    with open('{}{}.json'.format(path, save_weight_filename), 'w') as f:
        json.dump(training_history.history, f, indent=2)

    json_list = ['{}{}.json'.format(path, save_weight_filename)]
    for json_filename in json_list:
        with open(json_filename) as f:
            # convert the loss json object to a python dict
            loss_dict = json.load(f)
        print_list = ['loss', 'val_loss', 'dice_coef', 'val_dice_coef']
        for item in print_list:
            item_list = []
            if item in loss_dict:
                item_list.extend(loss_dict.get(item))
                plt.plot(item_list)
        plt.title('model:{} lr:{} epoch:{} #filtr:{} Colorspaces:{}'.format(model_name, learning_rate,
                                                                            epoch, n_filters, color_space))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss', 'test_loss', 'train_dice', 'test_dice'], loc='upper left')
        plt.savefig('{}{}.png'.format(path, save_weight_filename))
        plt.show()
        plt.clf()



