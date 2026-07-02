import torch
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity

from losses.ms_constants import KAPPA_EPS, MS_BASE_MARGIN, MS_NEG_WEIGHT, MS_POS_WEIGHT
from losses.vmf_loss import VMFLikelihood

def _basic_loss(loss_name):
    if loss_name == "SupConLoss":
        return losses.SupConLoss(temperature=0.07)
    if loss_name == "CircleLoss":
        return losses.CircleLoss(m=0.4, gamma=80)
    if loss_name == "MultiSimilarityLoss":
        return losses.MultiSimilarityLoss(
            alpha=MS_POS_WEIGHT,
            beta=MS_NEG_WEIGHT,
            base=MS_BASE_MARGIN,
            distance=DotProductSimilarity(),
        )
    if loss_name == "ContrastiveLoss":
        return losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if loss_name == "Lifted":
        return losses.GeneralizedLiftedStructureLoss(
            neg_margin=0, pos_margin=1, distance=DotProductSimilarity()
        )
    if loss_name == "FastAPLoss":
        return losses.FastAPLoss(num_bins=30)
    if loss_name == "NTXentLoss":
        return losses.NTXentLoss(temperature=0.07)
    if loss_name == "TripletMarginLoss":
        return losses.TripletMarginLoss(
            margin=0.1, swap=False, smooth_loss=False, triplets_per_anchor="all"
        )
    if loss_name == "CentroidTripletLoss":
        return losses.CentroidTripletLoss(
            margin=0.05, swap=False, smooth_loss=False, triplets_per_anchor="all"
        )
    raise NotImplementedError(f"Sorry, <{loss_name}> loss function is not implemented!")


def get_loss(loss_name, active_losses, uncertainty_loss, descriptors_dimension=512):
    """
    Return (loss_basic, loss_uncertainty) for MixVPR training.

    active_losses: e.g. ["basic"], ["uncertainty"], or both (from --losses).
    uncertainty_loss: "vmf" -> VMFLikelihood; "kappa_ms" -> joint Margin-MS + R(kappa).
    """

    loss_basic = None
    loss_uncertainty = None
    u = str(uncertainty_loss).lower()

    if u == "kappa_ms":
        if "uncertainty" not in active_losses:
            raise ValueError("kappa_ms requires --losses uncertainty")
        from losses.kappa_ms_loss import KappaMSLoss

        loss_uncertainty = KappaMSLoss(d=int(descriptors_dimension))
        return loss_basic, loss_uncertainty

    if "basic" in active_losses:
        loss_basic = _basic_loss(loss_name)

    if "uncertainty" in active_losses:
        if u == "vmf":
            loss_uncertainty = VMFLikelihood(d=int(descriptors_dimension))
        elif u == "gaussian_nll":
            loss_uncertainty = torch.nn.GaussianNLLLoss()
        else:
            raise NotImplementedError(f"Unknown uncertainty_loss: {uncertainty_loss}")

    return loss_basic, loss_uncertainty


def get_miner(miner_name, margin=0.1):
    if miner_name == "TripletMarginMiner":
        return miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard")
    if miner_name == "MultiSimilarityMiner":
        return miners.MultiSimilarityMiner(epsilon=margin, distance=CosineSimilarity())
    if miner_name == "PairMarginMiner":
        return miners.PairMarginMiner(
            pos_margin=0.7, neg_margin=0.3, distance=DotProductSimilarity()
        )
    return None
