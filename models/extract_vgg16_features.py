import json
import os
from argparse import ArgumentParser

import numpy as np
from keras.applications import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.preprocessing import image

from utils import Progbar

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--img_names_filename", type=str, required=True)
    parser.add_argument("--img_features_filename", type=str, required=True)
    args = parser.parse_args()

    img_filenames = os.listdir(args.img_path)
    num_img_filenames = len(img_filenames)
    img_features = []
    progress = Progbar(num_img_filenames)

    base_model = VGG16(weights="imagenet")
    model = Model(input=base_model.input, output=base_model.get_layer("fc2").output)

    for num_filename, filename in enumerate(img_filenames, 1):
        if os.path.splitext(filename)[-1].lower() == ".jpg":
            img = image.load_img(os.path.join(args.img_path, filename), target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            fc7_features = model.predict(x).squeeze()
            img_features.append(fc7_features)
            progress.update(num_filename)

    img_features = np.array(img_features)

    with open(args.img_names_filename, "w") as out_file:
        json.dump(img_filenames, out_file)

    np.save(args.img_features_filename, img_features)
