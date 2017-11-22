import json
import os
from argparse import ArgumentParser

import numpy as np
from keras import Model
from keras.applications import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image


def extract_image_features(images_path):
    base_model = VGG16(weights="imagenet")
    model = Model(input=base_model.input, output=base_model.get_layer("fc2").output)
    img_names = []
    img_features = []
    files = os.listdir(images_path)
    for file_index, file in enumerate(files):
        if os.path.splitext(file)[-1].lower() == ".jpg":
            img = image.load_img(os.path.join(images_path, file), target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            fc7_features = model.predict(x).squeeze()
            print("Reading file {} [{}/{}]".format(file, file_index, len(files)))
            img_names.append(file)
            img_features.append(fc7_features)

    img_features = np.array(img_features)

    return img_names, img_features


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--img_names_filename", type=str, required=True)
    parser.add_argument("--img_features_filename", type=str, required=True)
    args = parser.parse_args()
    img_names, img_features = extract_image_features(args.img_path)
    with open(args.img_names_filename, "w") as in_file:
        json.dump(img_names, in_file)
    np.save(args.img_features_filename, img_features)
