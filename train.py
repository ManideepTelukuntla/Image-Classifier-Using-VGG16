import torch
import argparse
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from functions import load_data, construct_classifier, validator, train_model, test_model, save_model, loading_model

parser = argparse.ArgumentParser(description = 'Train deep neural network.')

# ../aipnd-project/flowers
parser.add_argument('data_directory', action = 'store',
                    help = 'Enter path to the data.')

parser.add_argument('--arch', action = 'store',
                    dest = 'pretrained_model', default = 'vgg16',
                    help = 'Enter pretrained model to use, default is VGG-16, you can try VGG-13')

parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_directory', default = 'checkpoint.pth',
                    help = 'Enter location to save checkpoint in.')

parser.add_argument('--epochs', action = 'store',
                    dest = 'num_epochs', type = int, default = 2,
                    help = 'Enter number of epochs to use during training, default is 2.')

parser.add_argument('--learning_rate', action = 'store',
                    dest = 'lr', type = float, default = 0.001,
                    help = 'Enter learning rate for training the model, default is 0.001.')

parser.add_argument('--dropout', action = 'store',
                    dest = 'drpt', type = float, default = 0.2,
                    help = 'Enter dropout for training the model, default is 0.2.')


parser.add_argument('--hidden_units', action = 'store',
                    dest = 'units', type = int, default = 4096,
                    help = 'Enter number of hidden units in classifier, default is 4096.')


parser.add_argument('--gpu', action = "store_true", default = False,
                    help = 'Turn GPU mode on or off, default is off.')


results = parser.parse_args()

data_dir = results.data_directory
save_dir = results.save_directory
learning_rate = results.lr
dropout = results.drpt
hidden_units = results.units
epochs = results.num_epochs
pre_train_model = results.pretrained_model

# Set device(gpu or cpu)
if results.gpu == True:
    device = 'cuda'
else:
    device = 'cpu'

# Load data
train_loader, validation_loader, test_loader, train_data, validation_data, test_data = load_data(data_dir)

# Load pretrained model
model = getattr(models,pre_train_model)(pretrained=True)

# Build new classifier
input_units = model.classifier[0].in_features
construct_classifier(model, input_units, hidden_units, dropout)

# Define criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

# Train model
model, optimizer = train_model(model, epochs, train_loader, validation_loader, criterion, optimizer, device)

# Test model
test_model(model, test_loader, criterion, device)

# Save model
save_model(model, save_dir, epochs, optimizer, train_data)
