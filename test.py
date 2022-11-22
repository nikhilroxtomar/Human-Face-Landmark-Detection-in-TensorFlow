
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from train import create_dir, load_dataset

global image_h
global image_w
global num_landmarks

def plot_lankmarks(image, landmarks):
    h, w, _ = image.shape
    radius = int(h * 0.005)

    for i in range(0, len(landmarks), 2):
        x = int(landmarks[i] * w)
        y = int(landmarks[i+1] * h)

        image = cv2.circle(image, (x, y), radius, (255, 0, 0), -1)

    return image

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Hyperparameters """
    image_h = 512
    image_w = 512
    num_landmarks = 106

    """ Paths """
    dataset_path = "/media/nikhil/Seagate Backup Plus Drive/ML_DATASET/LaPa"
    model_path = os.path.join("files", "model.h5")

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")

    """ Load the model """
    model = tf.keras.models.load_model(model_path)
    # model.summary()

    """ Prediction """
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image_x = image
        image = cv2.resize(image, (image_w, image_h))
        image = image/255.0 ## (512, 512, 3)
        image = np.expand_dims(image, axis=0) ## (1, 512, 512, 3)
        image = image.astype(np.float32)

        """ Landmarks """
        data = open(y, "r").read()
        landmarks = []
        for line in data.strip().split("\n")[1:]:
            x, y = line.split(" ")
            x = float(x)/image_x.shape[1]
            y = float(y)/image_x.shape[0]

            landmarks.append(x)
            landmarks.append(y)

        landmarks = np.array(landmarks, dtype=np.float32)

        """ Prediction """
        pred = model.predict(image, verbose=0)[0]
        pred = pred.astype(np.float32)

        """ Saving the results """
        gt_landmarks = plot_lankmarks(image_x.copy(), landmarks)
        pred_landmarks = plot_lankmarks(image_x.copy(), pred)
        line = np.ones((image_x.shape[0], 10, 3)) * 255

        cat_images = np.concatenate([gt_landmarks, line, pred_landmarks], axis=1)
        cv2.imwrite(f"results/{name}.png", cat_images)






    ## ...
