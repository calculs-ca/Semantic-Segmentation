# Image Semantic-Segmentation
Multi-class Image Semantic Segmentation on underwater dataset.
There are two implemented models: a simple convolutional model and a Unet.
The training uses *Pytorch Lightning* and the hyperparameter optimization is done with *Optuna*.
The metrics, code and visualization are logged to Comet.ml during training. Check [comet.ml | semantic-segmentation](https://www.comet.ml/aklopezcarbajal/semantic-segmentation/view/XTQcsRTTflYLoZIqMLn7mpZc6/panels).

## Dataset
[Semantic Segmentation of Underwater Imagery SUIM](https://www.kaggle.com/ashish2001/semantic-segmentation-of-underwater-imagery-suim)

This dataset contains over 1500 images with pixel annotations for eight object categories: fish (vertebrates), reefs
(invertebrates), aquatic plants, wrecks/ruins, human divers, robots, and sea-floor.

## Setup
* Create virtual environment
* Install requirements
```
pip install -r requirements.txt 
```
* Activate environment

## Pre-process data
The transformed data is stored in a ```.pt``` file. To generate said file run the code once passing the flag ```preprocess=True``` to ```prepare_data()```. 
Make sure you set the environment variable ```DATA_PATH``` with the directory of the dataset.
```
export DATA_PATH=/path/to/data
```
