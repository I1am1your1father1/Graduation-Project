import torch
from tqdm import tqdm

def loss_mis_qubo(outs: torch.Tensor, Q, **kwargs):
    """
    Loss function for mis problem.
    """

    epoch = kwargs.get("epoch", 1)

    x = outs[:, 0]
    loss_obj = x.T @ Q @ x

    if epoch % 100 == 0:
        tqdm.write(f"Epoch: {epoch:.2f} | " f"obj Loss: {loss_obj:.2f} ")

    return loss_obj