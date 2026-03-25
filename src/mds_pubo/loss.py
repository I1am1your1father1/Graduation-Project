from tqdm import tqdm
import torch

def loss_mds_onehot_pubo(outs: torch.Tensor, H: torch.Tensor, **kwargs):
    r"""Loss function for Dominating Set (MDS) problem formulated as One-Hot PUBO (OH-PUBO)"""

    epoch = kwargs.get("epoch", 1)
    num_epochs = kwargs.get("num_epochs", None)
    cons_lambda = kwargs.get("cons_conf_lambda", lambda e, n: 1.0)
    obj_lambda = kwargs.get("obj_conf_lambda", lambda e, n: 1.0)

    cons_conf = cons_lambda(epoch, num_epochs)
    obj_conf = obj_lambda(epoch, num_epochs)

    X_ = outs.t().unsqueeze(-1)
    H_ = H.unsqueeze(0)
    mid = X_ * H_
    sub = (mid + (1 - H)).prod(dim=1).sum()  # Set the irrelevant position to 1 so that it cannot participate in multiplication

    loss_cons_obj = sub
    loss_obj = (- outs).sum()

    if epoch % 100 == 0:
        tqdm.write(f"Epoch: {epoch:.2f} | " f"obj Loss: {loss_obj:.2f} | " f"cons Loss: {loss_cons_obj:.2f} |")
   
    loss_cons_obj = 2 * cons_conf * loss_cons_obj
    loss_obj = obj_conf * loss_obj

    return loss_obj + loss_cons_obj
