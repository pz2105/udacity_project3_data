#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

from PIL import ImageFile






NUM_CLASS = 133


def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    print("Testing Model on Whole Testing Dataset")
    model.eval() # set the evaluation mode
    running_loss=0
    running_corrects=0
    
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    print(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")
    return 100*total_acc



def train(model, train_loader, criterion, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    for epoch in range(1, epochs + 1):
        model.train() # set the training mode
        running_loss = 0.0
        running_corrects = 0
        running_samples=0
        

        for step, (inputs, labels) in enumerate(train_loader):
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            running_samples+=len(inputs)
            if running_samples % 20  == 0:
                accuracy = running_corrects/running_samples
                print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                        running_samples,
                        len(train_loader.dataset),
                        100.0 * (running_samples / len(train_loader.dataset)),
                        loss.item(),
                        running_corrects,
                        running_samples,
                        100.0*accuracy,
                    )
                )
                
    return model

    
    
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    # we will be using resnet 50, freezes parameters, and add two more layers.
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, NUM_CLASS*2),
                   nn.ReLU(inplace=True),
                   nn.Linear(NUM_CLASS*2, NUM_CLASS))
    return model





def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    # data param is assume to be of dataset class
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)



def save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

    


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    # since we have 133 categories, we will use crossentropyloss, there will be lr and momentum
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    ## using transform to get data from s3 and convert them into dataloader
    ## class-----------
        ## notice we have downloaded data from online, and unzip the data into 
        ## local folders, so we can directly use them.
    train_transforms = transforms.Compose([  # skipping normalization
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True # this is necessary because we transformed images
    
    # get the directory of dataset
    TRAIN_DATASET_PATH = os.path.join(args.data, 'dogImages/train/')
    VALID_DATASET_PATH = os.path.join(args.data, 'dogImages/valid/')

    train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_DATASET_PATH,
                                                     transform=train_transforms)
    valid_dataset = torchvision.datasets.ImageFolder(root=VALID_DATASET_PATH,
                                                     transform=valid_transforms)

    # call create_data_loaders method to convert dataset class to dataloader class
    train_loader = create_data_loaders(train_dataset, batch_size=args.batch_size)
    valid_loader = create_data_loaders(valid_dataset, batch_size=args.batch_size)
    

    ## end of converting data--------------
    
    
#     if args.gpu == 1:
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
    device = torch.device("cpu")
    model.to(device)

    
    model=train(model, train_loader, loss_criterion, optimizer, args.epochs, device=device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, valid_loader, loss_criterion, device=device)
    
    '''
    TODO: Save the trained model
    '''
    save_model(model, args.model_dir)

    
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
#     parser.add_argument('--gpu', type = int, default=os.environ['SM_CHANNEL_GPU'])
    parser.add_argument('--data', type = str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)



