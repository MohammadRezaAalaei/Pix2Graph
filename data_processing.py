import torch
from torchvision import transforms
from Utils import Scaler, extract_type, extract_adjacencies, data_augmentation, Space_layout_dataset, PadCenterCrop
N_SAMPLES = 80000

adj_scaler = Scaler(5)
adj_scaler.filter.weight = torch.nn.Parameter(adj_scaler.scaler_filter)
adj_scaler.filter.bias = torch.nn.Parameter(torch.tensor([0.], dtype=torch.float))
front_scaler = Scaler(3)
front_scaler.filter.weight = torch.nn.Parameter(front_scaler.scaler_filter)
front_scaler.filter.bias = torch.nn.Parameter(torch.tensor([0.], dtype=torch.float))
node_scaler = Scaler(3)
node_scaler.filter.weight = torch.nn.Parameter(node_scaler.scaler_filter)
node_scaler.filter.bias = torch.nn.Parameter(torch.tensor([0.], dtype=torch.float))
random_xy = torch.load('random_xy.pt')
print(random_xy.shape)
scalers = {'front_scaler':front_scaler, 'node_scaler':node_scaler, 'adj_scaler':adj_scaler}
RPLAN_path = 'floorplan_dataset/'
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
    path=RPLAN_path,
    from_data_point=0, to_data_point=100,
    augmentation_type = None,
    transformation = None,
    random_xy=random_xy,
    step=step,
    scalers=scalers,
    type_extractor=extract_type,
    adjacency_extractor=extract_adjacencies
)
update(adj, coo, nodes, t_list, idx, mean_list, boundary_list)
adj, coo, nodes, t_list, idx, mean_list, boundary_list, step = data_augmentation(
    path=RPLAN_path,
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
    path=RPLAN_path,
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
    path=RPLAN_path,
    from_data_point=0, to_data_point=100,
    augmentation_type = 'crop',
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
data_g = Space_layout_dataset('data_g', features=g_features, edges=edges)
data_t = Space_layout_dataset('data_t', features=t_features, edges=edges)
data_idx = Space_layout_dataset('data_idx', features=index, edges=edges)
data_b = Space_layout_dataset('data_b', features=boundaries, edges=edges)
data_m = Space_layout_dataset('data_m', features=mean_features, edges=edges)
ix = torch.cat(index)
torch.save(ix, 'indexes.pt')


def number_of_nodes_per_graph(ix):
    """Given a flattened list of graph indices, returns the number of nodes per graph"""
    index_data = []
    for i in ix.unique():
        index_data.append(len(ix[ix == i]))
    return index_data
n_node_graph = number_of_nodes_per_graph(ix)
n_node_graph = torch.tensor(n_node_graph)
torch.save(n_node_graph, 'n_node_graph.pt')
