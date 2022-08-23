# Image Classifier

In this project, I trained an image classifier to recognize different species of flowers. I can imagine using something like this in a phone app that tells me the name of the flower my camera is looking at.I used [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories as part of this project.


## Prerequisites
* Python
* Jupyter Notebook (Use anaconda or miniconda to install jupyter notebook)
* GPU:
  * As the network makes use of a sophisticated deep convolutional neural network  the training process is impossible to be done by a common laptop. In order to train your models to your local machine you have three options.

    * **Cuda** -- If you have an NVIDIA GPU then you can install CUDA from [here](https://developer.nvidia.com/cuda-downloads). With Cuda you will be able to train your model however the process will still be time consuming
    * **Cloud Services** -- There are many paid cloud services that let you train your models like [AWS](https://aws.amazon.com/fr/) or  [Google Cloud](https://cloud.google.com/)
    * **Google Colab** -- [Google Colab](https://colab.research.google.com/) gives you free access to a GPU for limited time.
 * Additional files like ```cat_to_name.json(provided with repo)``` which is used to map labels into actual flower names.

## Implementation

### Part 1 - Jupyter Notebook Implementation
* In the first part of this project I have used jupyter notebook to implement an image classifier with PyTorch

### Part 2 - Building the Command Line Application
* Train a new network on a data set with ```train.py```
  * Basic Usage : ```python train.py data_directory```
  * Prints out current epoch, training loss, validation loss, and validation accuracy as the netowrk trains
  * Options:
    * Set direcotry to save checkpoints: ```python train.py data_dor --save_dir save_directory```
    * Choose arcitecture : ```pytnon train.py data_dir --arch "vgg16"```
    * Set hyperparameters: ```python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 ```
    * Use GPU for training: ```python train.py data_dir --gpu```

* Predict flower name from an image with ```predict.py``` along with the probability of that name. That is you'll pass in a single image ```/path/to/image``` and return the flower name and class probability
  * Basic usage: ```python predict.py /path/to/image checkpoint```
  * Options:
    * Return top **K** most likely classes:``` python predict.py input checkpoint ---top_k 3```
    * Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_To_name.json```
    * Use GPU for inference: ```python predict.py input checkpoint --gpu```
