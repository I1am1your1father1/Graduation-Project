from tqdm import tqdm
import torch


def loss_maxcut_gini_qubo(outs: torch.Tensor, Q, **kwargs):
    r"""Loss function for graph maxcut problem formulated as One-Hot QUBO (OH-QUBO)."""
    epoch = kwargs.get("epoch", 1)
    num_epochs = kwargs.get("num_epochs", None)
    gini_cof_lambda = kwargs.get("gini_cof_lambda", lambda e, n: 0)

    gini_cof = gini_cof_lambda(epoch, num_epochs)

    loss_gini = _gini_annealed_loss(outs)

    # loss_obj = (outs.t().mm(Q) * outs.t()).sum()  !!!!!!!!!!!!!!!Warning!!!!!!!!!!!!! x^2 != x
    Q_diag = Q.diag()
    Q_nodiag = Q.clone()
    Q_nodiag.fill_diagonal_(0)
    loss_obj = ((outs.T.mm(Q_nodiag) * outs.T) + (outs.T * Q_diag)).sum()

    if epoch % 100 == 0:
        tqdm.write(f"Epoch: {epoch:.2f} | " f"obj Loss: {loss_obj:.2f} | " f"annealing Loss: {loss_gini:.2f} | ")

    loss_gini = gini_cof * loss_gini

    return loss_gini + loss_obj


def _gini_annealed_loss(x: torch.Tensor):
    """Gini Coefficient-Based Annealing Algorithm"""
    gini = (1 - (2 * x - 1).pow(2)).sum()
    return gini

