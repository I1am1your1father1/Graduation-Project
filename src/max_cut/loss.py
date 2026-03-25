from tqdm import tqdm
import torch


def loss_maxcut_qubo(outs: torch.Tensor, Q, **kwargs):
    r"""Loss function for graph maxcut problem formulated as One-Hot QUBO (OH-QUBO)."""
    epoch = kwargs.get("epoch", 1)
    num_epochs = kwargs.get("num_epochs", None)

    # loss_obj = (outs.t().mm(Q) * outs.t()).sum()  !!!!!!!!!!!!!!!Warning!!!!!!!!!!!!! x^2 != x
    Q_diag = Q.diag()
    Q_nodiag = Q.clone()
    Q_nodiag.fill_diagonal_(0)
    loss_obj = ((outs.T.mm(Q_nodiag) * outs.T) + (outs.T * Q_diag)).sum()

    if epoch % 50 == 0:
        tqdm.write(f"Epoch: {epoch:.2f} | " f"obj Loss: {loss_obj:.2f} | ")

    return loss_obj
