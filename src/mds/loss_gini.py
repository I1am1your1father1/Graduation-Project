from tqdm import tqdm
import torch

def loss_mds_gini_qubo(outs: torch.Tensor, B=None, Q=None, **kwargs):
    r"""Loss function for Minimum Dominating Set (MDS) problem formulated as QUBO-style quadratic penalty"""

    epoch = kwargs.get("epoch", 1)
    num_epochs = kwargs.get("num_epochs", None)

    gini_cof_lambda = kwargs.get("gini_cof_lambda", lambda e, n: 1.0)
    cons_lambda = kwargs.get("cons_conf_lambda", lambda e, n: 1.0)
    obj_lambda = kwargs.get("obj_conf_lambda", lambda e, n: 1.0)

    gini_cof = gini_cof_lambda(epoch, num_epochs)
    cons_conf = cons_lambda(epoch, num_epochs)
    obj_conf = obj_lambda(epoch, num_epochs)

    if outs.dim() == 1:
        selected = outs
    elif outs.dim() == 2 and outs.size(1) == 1:
        selected = outs[:, 0]
    elif outs.dim() == 2 and outs.size(1) == 2:
        selected = outs[:, 1]
    else:
        raise ValueError(f"Unsupported outs shape for MDS-QUBO: {tuple(outs.shape)}")

    if B is None:
        raise ValueError("MDS-QUBO loss requires closed-neighborhood matrix B.")

    B = B.to(selected.device).float()

    # objective: minimize selected nodes
    loss_obj = selected.sum()

    # coverage[v] = sum of selected nodes in the closed neighborhood of v
    coverage = B @ selected

    # only penalize uncovered nodes
    violation = torch.relu(1.0 - coverage)
    loss_cons = violation.pow(2).sum()

    loss_gini = _gini_annealed_loss(outs)

    if epoch % 100 == 0:
        tqdm.write(
            f"Epoch: {epoch:.2f} | "
            f"obj Loss: {loss_obj:.2f} | "
            f"cons Loss: {loss_cons:.2f} | "
            f"annealing Loss: {loss_gini:.2f} | "
        )

    loss_cons = 2 * cons_conf * loss_cons
    loss_obj = obj_conf * loss_obj
    loss_gini = gini_cof * loss_gini

    return loss_obj + loss_cons + loss_gini


def _gini_annealed_loss(outs: torch.Tensor):
    """Gini Coefficient-Based Annealing Algorithm"""
    x = outs[:, 0]
    gini = (1 - (2 * x - 1).pow(2)).sum()

    return gini