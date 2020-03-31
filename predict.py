import json
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from torch import nn
from PIL import Image
from torch import optim
from workspace_utils import active_session
from torchvision import datasets, transforms, models
from functions import loading_model, predict, test_model, load_data, process_image

parser = argparse.ArgumentParser(description = 'Use neural network to make prediction on image.')

parser.add_argument('--image_path', action ='store',
                    default = 'flowers/test/13/image_05769.jpg',
                    help = "Enter path to image. Images are present in 'flowers/test/...'")

parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_directory', default = 'checkpoint.pth',
                    help = 'Enter location to save checkpoint in.')

parser.add_argument('--arch', action = 'store',
                    dest = 'pretrained_model', default='vgg16',
                    help = 'Enter pretrained model to use, default is VGG-16.')

parser.add_argument('--top_k', action = 'store',
                    dest = 'topk', type = int, default = 3,
                    help = 'Enter number of top most likely classes to view, default is 3.')

parser.add_argument('--category_names', action = 'store',
                    dest = 'cat_name_dir', default = 'cat_to_name.json',
                    help = 'Enter path to image.')

parser.add_argument('--gpu', action = "store_true", default = False,
                    help = 'Turn GPU mode on or off, default is off.')

results = parser.parse_args()

save_dir = results.save_directory
image_path = results.image_path
top_k = results.topk 
cat_names = results.cat_name_dir
pre_train_model = results.pretrained_model

# Set device(gpu or cpu)
if results.gpu == True:
    device = 'cuda'
else:
    device = 'cpu'

# Label mapping
with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

model = getattr(models,pre_train_model)(pretrained=True)

# Load model
loaded_model = loading_model(model, save_dir)

probabilities, classes = predict(image_path, loaded_model, top_k)

print(probabilities)
print(classes)

names = []
for i in classes:
    names += [cat_to_name[i]]

# Print name of predicted flower with highest probability
print(f"Flower is most likely to be a '{names[0]}' with a probability of {round(probabilities[0]*100,3)}% ")
