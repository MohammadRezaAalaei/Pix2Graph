import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Dataset, Data
import os
import os.path as osp
import numpy as np
import pandas as pd
import torchvision.transforms.functional as FF

kernel_size=20

class Scaler(nn.Module):
    def __init__(self, kernel_size = kernel_size):
        super(Scaler, self).__init__()
        self.scaler_filter =  torch.ones(size=[1,1,kernel_size,kernel_size], dtype=torch.float)
        self.filter = nn.Conv2d(1, 1, kernel_size, padding='same')

    def forward(self, x):
        x = self.filter(x)
        return x

# Transformation
class PadCenterCrop(object):
    def __init__(self, size, pad_if_needed=True, fill=256, padding_mode='edge'):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode
        self.fill = fill

    def __call__(self, img):

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = FF.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = FF.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        return FF.center_crop(img, self.size)


def extract_type(node, g_list):
    """
    This function extracts the type of each room
    arguments:
        node: tensor of size (resolution, resolution) indicating the current node
        g_list: tensor of size (n_spaces, resolution, resolution), list of spaces seperated by their type
    return:
        A one hot encoded pytorch tensor indicating the type of current node
    """
    typ = torch.zeros(len(g_list))
    for i in range(len(g_list)):
        subtraction = g_list[i] - node
        subtraction[subtraction < 0] = 0
        if torch.sum(g_list[i]) - torch.sum(subtraction):
            typ[i] = 1
            break

    return typ, node


def extract_adjacencies(node, nodes, adj_scaler):
    """
    This function extracts the adjacencies of each room
    arguments:
        node: tensor of size (resolution, resolution) indicating the current node
        nodes: tensor of size (n_spaces, resolution, resolution) including all nodes
    returns:
        An adjacency list including 1 for adjacent spaces, else 0
    """
    scaled = []
    node = adj_scaler(node.view(1, 1, node.shape[0], node.shape[1])).detach().squeeze()
    node[node != 0] = 1

    adj_nodes = []
    for i in nodes:
        if torch.sum(torch.stack([node, i]), dim=0).max() > 1:
            adj_nodes.append(1)
        else:
            adj_nodes.append(0)

    return adj_nodes



def data_augmentation(from_data_point=0,
                      to_data_point=1000,
                      augmentation_type=None,
                      transformation= None,
                      step=None,
                      random_xy=None,
                      scalers=None,
                      type_extractor=None,
                      adjacency_extractor=None):
    global g_features, t_features, mean_features, edges, index, adjacency_matrix, boundaries, entrance
    node_scaler, zone_scaler, adj_scaler = scalers['node_scaler'], scalers['zone_scaler'], scalers['adj_scaler']
    for i, (j1, j2) in tqdm(enumerate(random_xy[from_data_point:to_data_point])):
        img = Image.open(f'floorplan_dataset/{i}.png')
        if augmentation_type == 'flip':
            data_transforms = transforms.Compose([
                transformation,
            ])
            img = data_transforms(img)
        elif augmentation_type == 'crop':
            centercrop = PadCenterCrop([j1.item(), j2.item()])
            img = centercrop(img)

        convert_tensor = transforms.ToTensor()
        tensor = convert_tensor(img)
        tensor = F.interpolate(tensor.view(1, tensor.shape[0], tensor.shape[1], tensor.shape[2]),size=[64,64]).squeeze()
        boundary = tensor[3,:, :]
        types = tensor[1, :, :]
        indexes = tensor[2, :, :]
        nodes = [torch.zeros_like(indexes) for i in range(len(indexes.unique())-1)]
        living_room = torch.zeros_like(types)
        master_room = torch.zeros_like(types)
        kitchen = torch.zeros_like(types)
        bathroom = torch.zeros_like(types)
        dining_room = torch.zeros_like(types)
        second_room = torch.zeros_like(types)
        balcony = torch.zeros_like(types)
        storage = torch.zeros_like(types)
        front_door = torch.zeros_like(types)

        living_room[(types>-0.5/256) & (types<0.5/256)] = 1
        master_room[(types>0.5/256) & (types<1.5/256)] = 1
        kitchen[(types>1.5/256) & (types<2.5/256)] = 1
        bathroom[(types>2.5/256) & (types<3.5/256)] = 1
        dining_room[(types>3.5/256) & (types<4.5/256)] = 1
        second_room[(types>4.5/256) & (types<5.5/256)] = 1
        second_room[(types>5.5/256) & (types<6.5/256)] = 1
        second_room[(types>6.5/256) & (types<7.5/256)] = 1
        second_room[(types>7.5/256) & (types<8.5/256)] = 1
        balcony[(types>8.5/256) & (types<9.5/256)] = 1
        storage[(types>10.5/256) & (types<11.5/256)] = 1
        front_door[(types>14.5/256) & (types<15.5/256)] = 1
        if not torch.sum(front_door):
            continue
        g_temp_list = [living_room, master_room, kitchen, bathroom, dining_room, second_room,
                 balcony, storage, front_door]

        g_list = []
        t_list = []
        mean_list = []
        adj = []
        for j in range(len(nodes)):
            nodes[j][indexes==indexes.unique()[j+1]] = 1
            typ, node = type_extractor(nodes[j], g_temp_list)
            arg_x, arg_y = torch.where(node!=0)
            mean_x = torch.mean(arg_x.type(torch.FloatTensor))
            mean_y = torch.mean(arg_y.type(torch.FloatTensor))
            mean_x = mean_x.round().type(torch.LongTensor)
            mean_y = mean_y.round().type(torch.LongTensor)
            noise_factor = 3
            noise_x = torch.randint(low=-noise_factor,high=noise_factor, size=[1])
            noise_y = torch.randint(low=-noise_factor,high=noise_factor, size=[1])
            mean_x = min(max(arg_x.min(), mean_x + noise_x), arg_x.max())
            mean_y = min(max(arg_y.min(), mean_y + noise_y), arg_y.max())
            mean = torch.zeros_like(node)
            mean[mean_x, mean_y] = 1
            mean = node_scaler(mean.view(1, 1, 64, 64)).squeeze().detach()
            mean[mean!=0] = 1
            if typ.sum():
                t_list.append(torch.tensor(typ))
                g_list.append(torch.tensor(node))
                mean_list.append(mean)

        for node in range(len(nodes)):
            node_adj = adjacency_extractor(nodes[node], nodes, adj_scaler)
            adj.append(torch.tensor(node_adj))
            #         Visualization of adjacencies
    #         node_adj = torch.tensor(node_adj)
    #         node_adj = node_adj.view(node_adj.shape[0], 1, 1) * torch.stack(nodes)
    #         node_adj = torch.sum(node_adj, dim=0)
    #         node_adj = node_adj + nodes[node]
    #         plt.imshow(nodes[node])
    #         plt.show()
    #         plt.imshow(node_adj)
    #         plt.show()

        adj = torch.stack(adj)
        coo = (adj > 0).nonzero().t()
        if adj.shape[0] != len(g_list):
            continue
        front_door = zone_scaler(front_door.view(1, 1, 64, 64)).squeeze()
        boundary[front_door!=0] = 0.3
        boundary_list = [boundary.clone() for _ in range(len(g_list))]
        boundary_list = torch.stack(boundary_list)
        mean_list = torch.stack(mean_list)
        t_list = torch.stack(t_list)
        nodes = torch.stack(g_list)
        idx = torch.ones(len(nodes)) * step
        step += 1
        return adj, coo, nodes, t_list, idx, mean_list, boundary_list, step




class Space_layout_dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, features=None):
        self.features = features
        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['node_features.pt', 'edge_features.pt']

    @property
    def processed_file_names(self):
        return [f'data{i}.pt' for i in range(len(self.features))]

    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        pass

    def process(self):
        # Read data into huge `Data` list.
        # node_geometrical_features.shape = n_samples, n_spaces, resolution, resolution
        # edge_features.shape = n_samples, n_spaces, n_spaces
        for i in range(len(index)):
            data = Data(
                x=self.features[i],
                edge_index=edges[i]
            )
            data = data.pin_memory()
            torch.save(data, os.path.join(self.processed_dir,
                                          f'data{i}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx=None, return_all=False):
        if return_all:
            data = [self.get(i) for i in range(self.len())]
            return data
        data = torch.load(osp.join(self.processed_dir, f'data{idx}.pt'))
        return data