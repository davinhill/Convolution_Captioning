# Convolution_Captioning

## Overview

Implementation of Convolutional Image Captioning (2018) by J. Aneja, A. Deshpande, and A. G. Schwing. https://arxiv.org/abs/1711.09151

Please refer to **final_report.pdf** for details on the motivation and results of this project.


## File Descriptions

### Training

For training, we need to run the **train.py** file. Please use the option --help to see the list of arguments.

For loading partially (or fully) trained models, please use the options --load_model and --accy_file to input the paths to the saved model and accuracy file.


### Testing

The **test_w_beamsearch** file takes similar inputs as the train.py file, and will ouput the test results for a specified number of beams. Please note that you must have the same arguments when loading the model as when the model was trained (so that the saved states can be loaded correctly)

### Other Files / Folders

**models.py** Code for the Convolution Captioning model class.

**img_encoders.py** Code for the 3 different image encoders implemented (VGG19, Resnet18, Densenet161)

**eval.py** Code for generating caption predictions and calculating the test accuracy (beam size == 1). Also contains functions for converting tokensIDs to words.

**dataloader.py** Code for the pytorch dataloader used in our model

**test_beam.py** Code for calling beamsearch functions

**beamsearch.py** Functions for beam search implementation. It receives the multiple beams and the log probs and selects the top beam_size. Cloned directly from https://github.com/aditya12agd5/convcap

The folder **embed** Includes files related to word embedding and preprocessing:
* **Create_word_embedding.ipynb** Jupyter Notebook for creating the tokenized vocabulary
* **preproc_glove.py** Script for calling downloading the GloVe pretrained word encoding and fitting it to our vocabulary
* **dict_to_df.py** Helper script for converting the model_accuracy.json dictionary file to a pandas dataframe (and saving as csv)



## Downloading Data

create folder 'coco_data2014' one level above the cloned repo.
inside this folder, download the data and unzip

```bash
wget http://images.cocodataset.org/zips/train2014.zip
unzip -q train2014.zip

wget http://images.cocodataset.org/zips/val2014.zip
unzip -q val2014.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip -q annotations.zip

wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip -q caption_datasets.zip
```

Sometimes the PycocoAPI has issues with the annotation file. In the annotations folder, create a new python script and run the following code.

```python
import json
with open('captions_val2014.json', 'r') as f:
    data = json.load(f)
    data['type'] = 'captions'
with open('captions_val2014.json', 'w') as f:
    json.dump(data, f)

with open('captions_train2014.json', 'r') as f:
    data = json.load(f)
    data['type'] = 'captions'
with open('captions_train2014.json', 'w') as f:
    json.dump(data, f)
```
