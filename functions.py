import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch import optim
from PIL import Image
from collections import OrderedDict
from torch.autograd import Variable
from workspace_utils import active_session
from torchvision import datasets, transforms, models


# Define function for loading image_datasets
def load_data(data_dir):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                                           transforms.RandomResizedCrop(224),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                                [0.229, 0.224, 0.225])])
    validation_test_transforms = transforms.Compose([transforms.Resize(245),
                                                                transforms.CenterCrop(224),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                                     [0.229, 0.224, 0.225])])


    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform = validation_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = validation_test_transforms)


    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size = 64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64)

    return train_loader, validation_loader, test_loader, train_data, validation_data, test_data



# Define function to build classifier
def construct_classifier(model, input_units, hidden_units, dropout):

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Define new classifier
    model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_units, hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('dropout', nn.Dropout(p = dropout)),
                                            ('fc2', nn.Linear(hidden_units, 102)),
                                            ('output', nn.LogSoftmax(dim = 1))]))

    return model



# Define function for validation
def validator(model, data_loader, criterion, device):
    model.to(device)

    loss = 0
    accuracy = 0

    for inputs, labels in data_loader:

        # Move parameters to the current device
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return loss, accuracy


# Define function to train the model
def train_model(model, epochs, train_loader, validation_loader, criterion, optimizer, device):
    model.to(device);
    steps = 0
    running_loss = 0
    print_every = 20
#     training_losses, validation_losses = [], []

    with active_session():
        # Train the network
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                steps += 1
             
                # Move parameters to the current device
                inputs, labels = inputs.to(device), labels.to(device)

                # Make sure previous gradients are erased
                optimizer.zero_grad()

                # Forward and backward pass
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Perform validation
                if steps % print_every == 0:
                    validation_loss = 0
                    accuracy = 0

                    # Set model to evaluation mode
                    model.eval()

                    # Turn off requires_grad
                    with torch.no_grad():

                        validation_loss, accuracy = validator(model, validation_loader, criterion, device)

#                     training_losses.append(running_loss / len(validation_loader))
#                     validation_losses.append(validation_loss / len(train_loader))

                    print(f"Epoch {epoch + 1}/{epochs}.. "
                          f"Train loss: {running_loss / print_every:.3f}.. "
                          f"Validation loss: {validation_loss / len(validation_loader):.3f}.. "
                          f"Validation accuracy: {accuracy / len(validation_loader):.3f}")

                    running_loss = 0

                    # Set model back to train mode
                    model.train()

        print("\nModel: I am ready to predict")

        return model, optimizer



# Define function to test the model
def test_model(model, test_loader, criterion, device):
    test_loss, accuracy = validator(model, test_loader, criterion, device)
    test_accuracy = accuracy / len(test_loader) * 100

    print(f"Test accuracy: {test_accuracy:.4f}%")



# Define function to save the model
def save_model(model, file_path, epochs, optimizer, train_data):

    checkpoint = {'state_dict': model.state_dict(),
                      'classifier': model.classifier,
                      'mapping': train_data.class_to_idx,
                      'opt_state': optimizer.state_dict,
                      'num_epochs': epochs}

    return torch.save(checkpoint, file_path)



# Dfinr function to load the model
def loading_model(model, file_path):

    checkpoint = torch.load(file_path)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']

    for param in model.parameters():
        param.requires_grad = False

    return model



# Define function to preprocess image_datasets
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    size = 224
    color_channels = 255

    # Open image
    original_image = Image.open(image)

    original_width, original_height = original_image.size


    if original_height > original_width:
        height = int(max(original_height * size / original_width, 1))
        width = int(size)
    else:
        width = int(max(original_width * size / original_height, 1))
        height = int(size)

    # Resize image
    original_image.thumbnail((width, height))

    # Crop image
    left = (width - size) / 2
    upper = (height - size) / 2
    right = left + size
    lower = upper + size
    cropped_image = original_image.crop((left, upper, right, lower))

    # Convert to numpy
    image_data = np.array(cropped_image) / color_channels

    # Normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image_array = (image_data - mean) / std

    # Set the color to the first channel and return
    return np_image_array.transpose(2, 0, 1)



# Define function to predict
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    model.eval()
    has_gpu_support = torch.cuda.is_available()
    model = model.cuda() if has_gpu_support else model.cpu()
    image_data = process_image(image_path)
    tensor = torch.from_numpy(image_data)

    tensor = tensor.float().cuda() if has_gpu_support else tensor
    with torch.no_grad():
        
        var_inputs = Variable(tensor)
        var_inputs = var_inputs.unsqueeze(0)
        output = model.forward(var_inputs)
        ps = torch.exp(output).data.topk(topk)

        probabilities = ps[0].cpu() if has_gpu_support else ps[0]
        classes = ps[1].cpu() if has_gpu_support else ps[1]

        class_to_idx_inverted = {model.class_to_idx[key]: key for key in model.class_to_idx}
        mapped_classes = [class_to_idx_inverted[label] for label in classes.numpy()[0]]
        probabilities = probabilities.numpy()[0]

    return probabilities, mapped_classes
