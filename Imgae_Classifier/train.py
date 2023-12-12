import numpy as np
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import json
from PIL import Image
import futility
import fmodel

def main(data_dir="./flowers/", save_dir="./checkpoint.pth", arch="vgg16", learning_rate=0.001, hidden_units=512, epochs=3, dropout=0.2, gpu="gpu"):
    where = data_dir
    path = save_dir
    lr = learning_rate
    struct = arch
    hidden_units = hidden_units
    power = gpu
    epochs = epochs
    dropout = dropout

    if torch.cuda.is_available() and power == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    trainloader, validloader, testloader, train_data = futility.load_data(where)
    model, criterion = fmodel.setup_network(struct,dropout,hidden_units,lr,power)
    optimizer = optim.Adam(model.classifier.parameters(), lr= 0.001)
    
    # Train Model
    steps = 0
    running_loss = 0
    print_every = 5
    print("--Training starting--")
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
          
            if torch.cuda.is_available() and power =='gpu':
                inputs, labels = inputs.to(device), labels.to(device)
                model = model.to(device)

            optimizer.zero_grad()

            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
        
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    
    model.class_to_idx =  train_data.class_to_idx
    torch.save({'structure' :struct,
                'hidden_units':hidden_units,
                'dropout':dropout,
                'learning_rate':lr,
                'no_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
    print("Saved checkpoint!")

if __name__ == "__main__":
    main()
