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

def main(input='./flowers/test/1/image_06752.jpg', data_dir="./flowers/", checkpoint='./checkpoint.pth', top_k=5, category_names='cat_to_name.json', gpu="gpu"):
    path_image = input
    number_of_outputs = top_k
    device = gpu
    json_name = category_names
    path = checkpoint

    model=fmodel.load_checkpoint(path)
    with open(json_name, 'r') as json_file:
        name = json.load(json_file)
        
    probabilities = fmodel.predict(path_image, model, number_of_outputs, device)
    probability = np.array(probabilities[0][0])
    labels = [name[str(index + 1)] for index in np.array(probabilities[1][0])]
    
    i = 0
    while i < number_of_outputs:
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1
    print("Finished Predicting!")

if __name__== "__main__":
    main()
