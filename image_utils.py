import sys
import scipy.io as sio
import json, itertools
import numpy as np
#from vgg16 import VGG16
#from keras.preprocessing import image
#from imagenet_utils import preprocess_input
import numpy as np
#from keras.models import Model


def_fnames='/users/ud2017/hoavt/data/flickr30k-cnn/flickr30k/filenames_77512.json'
def_feats='/users/ud2017/hoavt/data/flickr30k-cnn/flickr30k/vgg_feats_77512.npy'
images_path= '/users/ud2017/hoavt/data/flickr30k-images/'

def_fnames='/users/ud2017/hoavt/data/flickr8k/filenames_77512.json'
def_feats='/users/ud2017/hoavt/data/flickr8k/vgg_feats_77512.npy'
images_path= '/users/ud2017/hoavt/data/flickr8k/Flicker8k_Dataset/'

class ImageFeatures(object):
    def __init__(self, names_files=def_fnames, feats_files=def_feats):
        self.cache = {}
        self.name2idx={}
        self.names_files = names_files
        #self.feats_files = feats_files
        with open(self.names_files,'rb') as fn:
            self.names=json.load(fn)
        #print len(self.names)
        #fdata= sio.loadmat(feats_files)
        #self.feats = np.array(fdata.get('feats')).T
        #print self.feats.shape
        #for img_file, feat in itertools.izip(names,feats):
        #    self.cache[img_file] = feat
        #count=0
        self.feats = np.load(feats_files)
        for img_file in self.names:
            self.name2idx[img_file]= len(self.name2idx)
        
        #creat model
        if False:
            base_model = VGG16(weights='imagenet')
            self.model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

    def get_feat(self,img_file):
        if img_file in self.name2idx:
            return self.feats[self.name2idx[img_file]]
        return None
        return self.get_feat_model(img_file)
    
    def save_feat(self):
        np.save('/users/ud2017/hoavt/data/flickr30k-cnn/flickr30k/vgg_feats', self.feats)
        with open(self.names_files,'wb') as fn:
            json.dump(self.names, fn)
    def get_feat_model(self, img_file):
        print img_file
        img = image.load_img(images_path + img_file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        fc7_features = self.model.predict(x)
        self.names.append(img_file)
        self.name2idx[img_file] = len(self.name2idx)
        #self.feats.append(fc7_features)
        self.feats = np.vstack((self.feats, fc7_features))
        return fc7_features[0]

if __name__ == '__main__':
    images_feats = ImageFeatures()
    feats = images_feats.get_feat('152881593.jpg')
    print len(feats)
    print feats
