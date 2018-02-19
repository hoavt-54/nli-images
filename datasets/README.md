# Datasets

## Dataset for the task of recognizing whether a caption is a good (positive) or a bad (negative) caption for an image
The dataset generation procedure is described as follows. For each image contained in Foil, we sample:
- two positive captions from MSCOCO
- one negative caption from Foil
- one negative caption from MSCOCO such that is corresponds to an image with less or equal than two categories in common with the actual image
We consider Foil as a starting point because Foil is a subset of MSCOCO, so if we take images from Foil we can be sure that we can find negative captions both from Foil and from MSCOOCO for each image.

The script implementing the dataset generation procedure has been published [here](https://github.com/hoavt-54/nli-images/blob/master/models/build_ic_dataset.py).

The dataset generated using the described procedure is contained in the folder [IC](https://github.com/hoavt-54/nli-images/tree/master/datasets/IC).
