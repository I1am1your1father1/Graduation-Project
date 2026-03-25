from tqdm import tqdm
import torch

def loss_mds_gini_pubo(outs: torch.Tensor, closed_nbh, **kwargs):
    r"""Loss function for Dominating Set (MDS) problem formulated as One-Hot PUBO (OH-PUBO)"""

    epoch = kwargs.get("epoch", 1)
    num_epochs = kwargs.get("num_epochs", None)
    gini_cof_lambda = kwargs.get("gini_cof_lambda", lambda e, n: 1.0)
    cons_lambda = kwargs.get("cons_conf_lambda", lambda e, n: 1.0)
    obj_lambda = kwargs.get("obj_conf_lambda", lambda e, n: 1.0)

    gini_cof = gini_cof_lambda(epoch, num_epochs)
    cons_conf = cons_lambda(epoch, num_epochs)
    obj_conf = obj_lambda(epoch, num_epochs)

    unselected = outs[:, 0]
    selected = outs[:, 1]
    cons_terms = []
    for nbh in closed_nbh:
        cons_terms.append(unselected[nbh].prod())

    loss_cons = torch.stack(cons_terms).sum()
    loss_obj = selected.sum()
    loss_gini = _gini_annealed_loss(outs)

    if epoch % 100 == 0:
        tqdm.write(f"Epoch: {epoch:.2f} | " f"obj Loss: {loss_obj:.2f} | " f"cons Loss: {loss_cons:.2f} |" f"annealing  Loss: {loss_gini:.2f} | ")
   
    loss_cons = 2 * cons_conf * loss_cons
    loss_obj = obj_conf * loss_obj
    loss_gini = gini_cof * loss_gini

    return loss_obj + loss_cons + loss_gini


def _gini_annealed_loss(outs: torch.Tensor):
    """Gini Coefficient-Based Annealing Algorithm"""
    p = outs.pow(2).sum(dim=1)
    gini = (torch.ones_like(p) - p).sum()
    return gini
