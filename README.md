# Convolution_Captioning


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

in the annotations folder, create a new python script and run the following code. Otherwise it will have issues with the pycoco api

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
