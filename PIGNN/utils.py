import torch
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv
from itertools import chain, islice
from time import time


# GNN class to be instantiated with specified param values
class GCN_dev(nn.Module):
    def __init__(self, in_feats, hidden_size, number_classes, dropout, device):
        """
        Initialize a new instance of the core GCN model of provided size.
        Dropout is added in forward step.

        Inputs:
            in_feats: Dimension of the input (embedding) layer
            hidden_size: Hidden layer size
            dropout: Fraction of dropout to add between intermediate layer. Value is cached for later use.
            device: Specifies device (CPU vs GPU) to load variables onto
        """
        super(GCN_dev, self).__init__()

        self.dropout_frac = dropout
        self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.conv2 = GraphConv(hidden_size, number_classes).to(device)

    def forward(self, g, inputs):
        """
        Run forward propagation step of instantiated model.

        Input:
            self: GCN_dev instance
            g: DGL graph object, i.e. problem definition
            inputs: Input (embedding) layer weights, to be propagated through network
        Output:
            h: Output layer weights
        """

        # input step
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout_frac)

        # output step
        h = self.conv2(g, h)
        h = torch.sigmoid(h)

        return h


# Generate random graph of specified size and type,
# with specified degree (d) or edge probability (p)
def generate_graph(n, d=None, p=None, graph_type='reg', random_seed=0):
    """
    Helper function to generate a NetworkX random graph of specified type,
    given specified parameters (e.g. d-regular, d=3). Must provide one of
    d or p, d with graph_type='reg', and p with graph_type in ['prob', 'erdos'].

    Input:
        n: Problem size
        d: [Optional] Degree of each node in graph
        p: [Optional] Probability of edge between two nodes
        graph_type: Specifies graph type to generate
        random_seed: Seed value for random generator
    Output:
        nx_graph: NetworkX OrderedGraph of specified type and parameters
    """
    if graph_type == 'reg':
        print(f'Generating d-regular graph with n={n}, d={d}, seed={random_seed}')
        nx_temp = nx.random_regular_graph(d=d, n=n, seed=random_seed)
    elif graph_type == 'prob':
        print(f'Generating p-probabilistic graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.fast_gnp_random_graph(n, p, seed=random_seed)
    elif graph_type == 'erdos':
        print(f'Generating erdos-renyi graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.erdos_renyi_graph(n, p, seed=random_seed)
    else:
        raise NotImplementedError(f'!! Graph type {graph_type} not handled !!')

    # Networkx does not enforce node order by default
    nx_temp = nx.relabel.convert_node_labels_to_integers(nx_temp)
    # Need to pull nx graph into OrderedGraph so training will work properly
    nx_graph = nx.OrderedGraph()
    nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
    nx_graph.add_edges_from(nx_temp.edges)
    return nx_graph


# helper function to convert Q dictionary to torch tensor
def qubo_dict_to_torch(nx_G, Q, torch_dtype=None, torch_device=None):
    """
    Output Q matrix as torch tensor for given Q in dictionary format.

    Input:
        Q: QUBO matrix as defaultdict
        nx_G: graph as networkx object (needed for node lables can vary 0,1,... vs 1,2,... vs a,b,...)
    Output:
        Q: QUBO as torch tensor
    """

    # get number of nodes
    n_nodes = len(nx_G.nodes)

    # get QUBO Q as torch tensor
    Q_mat = torch.zeros(n_nodes, n_nodes)
    for (x_coord, y_coord), val in Q.items():
        Q_mat[x_coord][y_coord] = val

    if torch_dtype is not None:
        Q_mat = Q_mat.type(torch_dtype)

    if torch_device is not None:
        Q_mat = Q_mat.to(torch_device)

    return Q_mat


# Chunk long list
def gen_combinations(combs, chunk_size):
    yield from iter(lambda: list(islice(combs, chunk_size)), [])


# helper function for custom loss according to Q matrix
def loss_func(probs, Q_mat):
    """
    Function to compute cost value for given probability of spin [prob(+1)] and predefined Q matrix.

    Input:
        probs: Probability of each node belonging to each class, as a vector
        Q_mat: QUBO as torch tensor
    """

    probs_ = torch.unsqueeze(probs, 1)

    # minimize cost = x.T * Q * x
    cost = (probs_.T @ Q_mat @ probs_).squeeze()

    return cost


# Construct graph to learn on
def get_gnn(n_nodes, gnn_hypers, opt_params, torch_device, torch_dtype):
    """
    Generate GNN instance with specified structure. Creates GNN, retrieves embedding layer,
    and instantiates ADAM optimizer given those.

    Input:
        n_nodes: Problem size (number of nodes in graph)
        gnn_hypers: Hyperparameters relevant to GNN structure
        opt_params: Hyperparameters relevant to ADAM optimizer
        torch_device: Whether to load pytorch variables onto CPU or GPU
        torch_dtype: Datatype to use for pytorch variables
    Output:
        net: GNN instance
        embed: Embedding layer to use as input to GNN
        optimizer: ADAM optimizer instance
    """
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    dropout = gnn_hypers['dropout']
    number_classes = gnn_hypers['number_classes']

    # instantiate the GNN
    net = GCN_dev(dim_embedding, hidden_dim, number_classes, dropout, torch_device)
    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)

    # set up Adam optimizer
    params = chain(net.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(params, **opt_params)
    return net, embed, optimizer


# Parent function to run GNN training given input config
def run_gnn_training(q_torch, dgl_graph, net, embed, optimizer, number_epochs, tol, patience, prob_threshold):
    """
    Wrapper function to run and monitor GNN training. Includes early stopping.
    """
    # Assign variable for user reference
    inputs = embed.weight

    prev_loss = 1.  # initial loss value (arbitrary)
    count = 0       # track number times early stopping is triggered

    # initialize optimal solution
    best_bitstring = torch.zeros((dgl_graph.number_of_nodes(),)).type(q_torch.dtype).to(q_torch.device)
    best_loss = loss_func(best_bitstring.float(), q_torch)

    t_gnn_start = time()

    # Training logic
    for epoch in range(number_epochs):

        # get logits/activations
        probs = net(dgl_graph, inputs)[:, 0]  # collapse extra dimension output from model

        # build cost value with QUBO cost function
        loss = loss_func(probs, q_torch)
        loss_ = loss.detach().item()

        # Apply projection
        bitstring = (probs.detach() >= prob_threshold) * 1
        if loss < best_loss:
            best_loss = loss
            best_bitstring = bitstring

        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, Loss: {loss_}')

        # early stopping check
        # If loss increases or change in loss is too small, trigger
        if (abs(loss_ - prev_loss) <= tol) | ((loss_ - prev_loss) > 0):
            count += 1
        else:
            count = 0

        if count >= patience:
            print(f'Stopping early on epoch {epoch} (patience: {patience})')
            break

        # update loss tracking
        prev_loss = loss_

        # run optimization with backpropagation
        optimizer.zero_grad()  # clear gradient for step
        loss.backward()        # calculate gradient through compute graph
        optimizer.step()       # take step, update weights

    t_gnn = time() - t_gnn_start
    print(f'GNN training (n={dgl_graph.number_of_nodes()}) took {round(t_gnn, 3)}')
    print(f'GNN final continuous loss: {loss_}')
    print(f'GNN best continuous loss: {best_loss}')

    final_bitstring = (probs.detach() >= prob_threshold) * 1

    return net, epoch, final_bitstring, best_bitstring


# =========================
# MaxCut functions
# =========================

def gen_q_dict_maxcut(nx_G):
    """
    Construct QUBO matrix for MaxCut.

    MaxCut objective:
        maximize sum_(i,j in E) [x_i (1 - x_j) + x_j (1 - x_i)]

    Equivalent minimization form:
        minimize sum_(i,j in E) [2 x_i x_j - x_i - x_j]

    x_i = 0/1 means node i belongs to one of the two partitions.
    """
    Q_dic = {}

    for u, v in nx_G.edges():
        Q_dic[(u, v)] = Q_dic.get((u, v), 0) + 2
        Q_dic[(u, u)] = Q_dic.get((u, u), 0) - 1
        Q_dic[(v, v)] = Q_dic.get((v, v), 0) - 1

    return Q_dic


def loss_func_maxcut(probs, Q_mat):
    """
    Loss function for MaxCut QUBO.

    The Q matrix is constructed as the minimization form of negative cut value.
    Therefore, minimizing this loss is equivalent to maximizing the cut value.
    """
    probs_ = torch.unsqueeze(probs, 1)
    cost = (probs_.T @ Q_mat @ probs_).squeeze()

    return cost


def postprocess_maxcut_bitstring(bitstring, nx_graph):
    """
    Calculate MaxCut value from a 0/1 bitstring.

    Args:
        bitstring: tensor or list, 0/1 assignment of nodes
        nx_graph: NetworkX graph

    Returns:
        cut_value: number of cut edges
        cut_ratio: cut_value / number_of_edges
        partition_0: nodes assigned to 0
        partition_1: nodes assigned to 1
    """
    if isinstance(bitstring, torch.Tensor):
        bitstring_list = bitstring.detach().cpu().long().view(-1).tolist()
    else:
        bitstring_list = list(bitstring)

    partition_0 = set()
    partition_1 = set()

    for node, value in enumerate(bitstring_list):
        if int(value) == 0:
            partition_0.add(node)
        else:
            partition_1.add(node)

    cut_value = 0
    for u, v in nx_graph.edges():
        if int(bitstring_list[u]) != int(bitstring_list[v]):
            cut_value += 1

    total_edges = nx_graph.number_of_edges()
    cut_ratio = cut_value / total_edges if total_edges > 0 else 0.0

    return cut_value, cut_ratio, partition_0, partition_1


def run_gnn_training_maxcut(
    q_torch,
    dgl_graph,
    net,
    embed,
    optimizer,
    number_epochs,
    tol,
    patience,
    prob_threshold
):
    """
    Run PI-GNN training for MaxCut.

    This function does not modify the original run_gnn_training.
    Difference:
        original run_gnn_training records best_bitstring according to continuous loss;
        this MaxCut version records best_bitstring according to discrete bitstring loss.
    """
    inputs = embed.weight

    prev_loss = 1.
    count = 0

    best_bitstring = torch.zeros(
        (dgl_graph.number_of_nodes(),)
    ).type(q_torch.dtype).to(q_torch.device)

    best_bitstring_loss = loss_func_maxcut(best_bitstring.float(), q_torch)
    best_continuous_loss = None

    t_gnn_start = time()

    for epoch in range(number_epochs):
        probs = net(dgl_graph, inputs)[:, 0]

        # continuous relaxation loss
        loss = loss_func_maxcut(probs, q_torch)
        loss_ = loss.detach().item()

        # discrete solution
        bitstring = (probs.detach() >= prob_threshold) * 1
        bitstring = bitstring.type(q_torch.dtype).to(q_torch.device)

        # select best according to discrete QUBO loss
        bitstring_loss = loss_func_maxcut(bitstring.float(), q_torch)

        if bitstring_loss < best_bitstring_loss:
            best_bitstring_loss = bitstring_loss
            best_bitstring = bitstring.clone().detach()

        if best_continuous_loss is None or loss < best_continuous_loss:
            best_continuous_loss = loss.clone().detach()

        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, Loss: {loss_}')

        if (abs(loss_ - prev_loss) <= tol) | ((loss_ - prev_loss) > 0):
            count += 1
        else:
            count = 0

        if count >= patience:
            print(f'Stopping early on epoch {epoch} (patience: {patience})')
            break

        prev_loss = loss_

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    t_gnn = time() - t_gnn_start

    print(f'GNN training (n={dgl_graph.number_of_nodes()}) took {round(t_gnn, 3)}')
    print(f'GNN final continuous loss: {loss_}')
    print(f'GNN best continuous loss: {best_continuous_loss}')
    print(f'GNN best discrete loss: {best_bitstring_loss}')

    final_bitstring = (probs.detach() >= prob_threshold) * 1
    final_bitstring = final_bitstring.type(q_torch.dtype).to(q_torch.device)

    return net, epoch, final_bitstring, best_bitstring