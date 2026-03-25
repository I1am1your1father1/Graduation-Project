import torch
from tqdm import tqdm

def loss_mis_gini_qubo(outs: torch.Tensor, Q, **kwargs):
    """
    Loss function for mis problem.
    """

    epoch = kwargs.get("epoch", 1)
    num_epochs = kwargs.get("num_epochs", None)
    gini_cof_lambda = kwargs.get("gini_cof_lambda", lambda e, n: 0)

    gini_cof = gini_cof_lambda(epoch, num_epochs)

    loss_gini = _gini_annealed_loss(outs)

    x = outs[:, 0]
    loss_obj = x.T @ Q @ x 

    if epoch % 100 == 0:
        tqdm.write(f"Epoch: {epoch:.2f} | " f"obj Loss: {loss_obj:.2f} | " f"gini Loss: {loss_gini:.2f} | ")

    loss_gini = gini_cof * loss_gini

    return loss_gini + loss_obj

def _gini_annealed_loss(outs: torch.Tensor):
    """Gini Coefficient-Based Annealing Algorithm"""
    x = outs[:, 0]
    gini = (1 - (2 * x - 1).pow(2)).sum()

    return gini