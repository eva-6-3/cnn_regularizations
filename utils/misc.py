import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary


def is_cuda(debug=True):
    cuda = torch.cuda.is_available()
    if debug:
        print("[INFO] Cuda Avaliable : ", cuda)
    return cuda


def get_device():
    use_cuda = is_cuda(debug=False)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("[INFO] device : ", device)
    return device


def set_seed(seed=1):
    cuda = is_cuda(debug=False)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    print(f"[INFO] seed set {seed}")


def show_random_images(data_loader):
    data, target  = next(iter(data_loader))
    grid_img = torchvision.utils.make_grid(data.cpu().detach())
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))


def show_model_summary(model, input_size=(1, 28, 28)):
    summary(model, input_size=input_size)

