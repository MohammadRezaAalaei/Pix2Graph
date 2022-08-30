import torch
from torchvision import transforms
from Utils import Scaler, extract_type, extract_adjacencies, data_augmentation, Space_layout_dataset, PadCenterCrop
N_SAMPLES = 80000

adj_scaler = Scaler(5)
adj_scaler.filter.weight = torch.nn.Parameter(adj_scaler.scaler_filter)
adj_scaler.filter.bias = torch.nn.Parameter(torch.tensor([0.], dtype=torch.float))
zone_scaler = Scaler(3)
zone_scaler.filter.weight = torch.nn.Parameter(zone_scaler.scaler_filter)
zone_scaler.filter.bias = torch.nn.Parameter(torch.tensor([0.], dtype=torch.float))
node_scaler = Scaler(3)
node_scaler.filter.weight = torch.nn.Parameter(node_scaler.scaler_filter)
node_scaler.filter.bias = torch.nn.Parameter(torch.tensor([0.], dtype=torch.float))
random_xy = torch.load('random_xy.pt')
print(random_xy.shape)
scalers = {'zone_scaler':zone_scaler, 'node_scaler':node_scaler, 'adj_scaler':adj_scaler}

g_features = []
t_features = []
mean_features = []
edges = []
index = []
adjacency_matrix = []
boundaries = []
step = 0

def update(adj, coo, nodes, t_list, idx, mean_list, boundary_list):
    global g_features, t_features, mean_features, edges, adjacency_matrix, boundaries, index
    g_features.append(nodes)
    t_features.append(t_list)
    mean_features.append(mean_list)
    edges.append(coo)
    adjacency_matrix.append(adj)
    index.append(idx)
    boundaries.append(boundary_list)

adj, coo, nodes, t_list, idx, mean_list, boundary_list, step = data_augmentation(
    from_data_point=0, to_data_point=100,
    augmentation_type = 'flip',
    transformation = transforms.RandomHorizontalFlip(1),
    random_xy=random_xy,
    step=step,
    scalers=scalers,
    type_extractor=extract_type,
    adjacency_extractor=extract_adjacencies
)
update(adj, coo, nodes, t_list, idx, mean_list, boundary_list)
adj, coo, nodes, t_list, idx, mean_list, boundary_list, step = data_augmentation(
    from_data_point=0, to_data_point=100,
    augmentation_type = 'flip',
    transformation = transforms.RandomVerticalFlip(1),
    random_xy=random_xy,
    step=step,
    scalers=scalers,
    type_extractor=extract_type,
    adjacency_extractor=extract_adjacencies
)
update(adj, coo, nodes, t_list, idx, mean_list, boundary_list)
adj, coo, nodes, t_list, idx, mean_list, boundary_list, step = data_augmentation(
    from_data_point=0, to_data_point=100,
    augmentation_type = 'crom',
    transformation = None,
    random_xy=random_xy,
    step=step,
    scalers=scalers,
    type_extractor=extract_type,
    adjacency_extractor=extract_adjacencies
)
update(adj, coo, nodes, t_list, idx, mean_list, boundary_list)

torch.save(boundaries, 'boundary.pt')
torch.save(g_features, 'g_features.pt')
torch.save(t_features, 't_features.pt')
torch.save(index, 'index.pt')
torch.save(edges, 'edges.pt')
# data_g = Space_layout_dataset('F:\elmo sanat\thesis\جلسه 5\housGAN_dataset\dataset\.ipynb_checkpoints\project\data_g', features=g_features)
# data_t = Space_layout_dataset('data_t6', features=t_features)
# data_idx = Space_layout_dataset('data_idx6', features=index)
# data_b = Space_layout_dataset('data_b6', features=boundaries)
# data_m = Space_layout_dataset('data_m6', features=mean_features)
ix = torch.cat(index)
torch.save(ix, 'indexes.pt')

index_data = []
for i in ix.unique():
    index_data.append(len(ix[ix==i]))
index_data = torch.tensor(index_data)
torch.save(index_data, 'index_data6.pt')
