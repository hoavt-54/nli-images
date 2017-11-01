# SICK, SICK2, and VSICK2 datasets
The datasets, contained in the folders SICK, SICK2, and VSICK2, are described as follows:
* SICK/SICK_train_annotated.tsv, SICK/SICK_dev_annotated.tsv, SICK/SICK_test_annotated.tsv are the training, validation and test sets, respectively, generated from the SICK annotated dataset, whereas SICK/SICK_train.tsv, SICK/SICK_dev.tsv, SICK/SICK_test.tsv are the same sets without the annotations.
* SICK2/SICK2_annotated.tsv is the dataset generated from the SICK annotated dataset by considering only the pairs in which the premise is true in the image and in which the premise comes from Flickr, whereas SICK2/SICK2.tsv is the same set without the annotations. SICK2/difficult_SICK2_annotated.tsv, instead, is the dataset generated from SICK2/SICK2_annotated.tsv by considering only the most difficult pairs, whereas SICK2/SICK2.tsv is the same set without the annotations.
* VSICK2/VSICK2_annotated.tsv is the dataset generated from SICK2/SICK2_annotated.tsv by matching each pair with the respective image coming from Flickr, whereas VSICK2/VSICK2.tsv is the same set without the annotations. VSICK2/difficult_VSICK2_annotated.tsv, instead, is the dataset generated from VSICK2/VSICK2_annotated.tsv by considering only the most difficult pairs, whereas VSICK2/VSICK2.tsv is the same set without the annotations.

## Statistics of the datasets
The statistics of the datasets are reported in the following tables:

| dataset          | # ENTAILMENT | # CONTRADICTION | # NEUTRAL | # TOTAL |
|------------------|--------------|-----------------|-----------|---------|
| SICK2            | 669          | 306             | 1751      | 2726    |
| DIFFICULT SICK2  | 104          | 51              | 124       | 279     |
| VSICK2           | 682          | 309             | 1758      | 2749    |
| DIFFICULT VSICK2 | 106          | 52              | 122       | 280     |

| dataset          | # UNIQUE IMAGES |
|------------------|-----------------|
| VSICK2           | 990             |
| DIFFICULT VSICK2 | 249             |
