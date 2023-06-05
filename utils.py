import os 
from pathlib import Path 
from torch import save, nn
import torch

from PIL import Image
import matplotlib.pyplot as plt 

def save_model(model, dir:str): 
    if type(dir) == str: 
        dir = Path(dir)
    
    if not dir.exists(): 
        os.mkdir(dir)

    save(model.state_dict(), dir / f"{type(model).__name__}.pth")


def read_images_fromfolder(dir: str): 
    image_list = []
    if type(dir) == str: 
        dir = Path(dir)
    for fp in dir.iterdir(): 
        image_list.append(Image.open(fp))
    return image_list 

def plot_prob_distr(net, vocab, X:torch.Tensor):
    net.eval()
    probs = nn.Softmax(dim=1)(net(X)).sum(dim=0)
    class2prob = probs/probs.sum()
    class2prob = list(class2prob.cpu().detach().numpy())
    classes = vocab.t_encoder.inverse_transform(range(len(class2prob)))
    to_plot_data = sorted(list(zip(classes, class2prob)), key=lambda x: x[1])
    x, y = zip(*to_plot_data)
    plt.bar(x,y)
    plt.title('Prob of author:')
    plt.xticks(rotation=45);