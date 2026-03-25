import torch
from tqdm import tqdm


def loss_coloring_qubo(outs: torch.Tensor, Q: torch.Tensor, **kwargs):
    """Loss function for graph coloring problem formulated as One-Hot QUBO (OH-QUBO).

    The optimization goal is to minimize the number of colors used while ensuring valid graph coloring.

    Args:
        outs_cons (torch.Tensor): Assignment tensor of shape ``[V, K]`` representing vertex-color assignments. Each row
            is a one-hot vector. V indicates the number of vertices.
        outs_obj (torch.Tensor): Binary tensor of shape ``[1, K]`` indicating color usage where elements ∈ {0, 1}.
            K represents the maximum allowed colors (e.g., total number of vertices).
        Q (torch.Tensor): see `src/coloring/utils/coloring_construct_Q`.
        kwargs: Additional parameters including
            - epoch (int): Current training epoch.
            - num_epochs (int): Total number of training epochs.
            - gini_cof_lambda (Callable): Function to compute Gini coefficient weight,
                accepts (current_epoch, total_epochs).
            - cons_cof_lambda (Callable): Function to compute constraint weight,
                accepts (current_epoch, total_epochs).
            - obj_cof_lambda (Callable): Function to compute objective weight,
                accepts (current_epoch, total_epochs).
    """

    out_cons, out_obj = outs

    epoch = kwargs.get("epoch", 1)
    num_epochs = kwargs.get("num_epochs", None)

    cons_cof_lambda = kwargs.get("cons_cof_lambda", lambda e, n: 1.0)
    obj_cof_lambda = kwargs.get("obj_cof_lambda", lambda e, n: 1.0)
    obj_cons_cof_lambda = kwargs.get("obj_cons_cof_lambda", obj_cof_lambda)

    cons_cof = cons_cof_lambda(epoch, num_epochs)
    obj_cof = obj_cof_lambda(epoch, num_epochs)
    obj_cons_cof = obj_cons_cof_lambda(epoch, num_epochs)

    loss_cons_coloring = (out_cons.t().mm(Q) * out_cons.t()).sum()
    loss_obj = out_obj.sum()
    loss_cons_obj = ((torch.ones_like(loss_obj) - out_obj) * out_cons).sum()

    if epoch % 100 == 0:
        tqdm.write(
            f"Epoch: {epoch} | "
            f"coloring loss: {loss_cons_coloring.item():.2f} | "
            f"group loss: {loss_obj.item():.2f} | "
            f"obj cons loss: {loss_cons_obj.item():.2f} | "
        )

    loss_cons_coloring = cons_cof * loss_cons_coloring
    loss_obj = obj_cof * loss_obj
    loss_cons_obj = obj_cons_cof * loss_cons_obj
    
    return loss_cons_coloring + loss_obj + loss_cons_obj