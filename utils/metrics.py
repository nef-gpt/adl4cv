from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import torch
from einops import rearrange

ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none")


# image similarity metric
def ssim_distance(img1, img2):
    """
    Compute the Structural Similarity Index (SSIM) between grayscale MNIST images.

    :param img1: tensor of shape (28, 28) with values in [0, 1]
    :param img2: tensor of shape (28, 28) with values in [0, 1]

    :return: SSIM distance between img1 and img2 -> torch.Tensor of shape (B?, 1) * 10 ^ 2
    """
    return ssim(img1, img2) * 10**2


def pairwise_ssim_distance(imgs1, imgs2):
    """
    Compute the Structural Similarity Index (SSIM) between grayscale MNIST images.

    :param imgs1: tensor of shape (B1, 1, 28, 28) with values in [0, 1]
    :param imgs2: tensor of shape (B2, 1, 28, 28) with values in [0, 1]

    :return: Matrix of SSIM distances between imgs1 and imgs2 -> torch.Tensor of shape (B1, B2)
    """
    B1 = imgs1.size(0)
    B2 = imgs2.size(0)

    # TODO: compare every image in imgs1 with every image in imgs2
    distances = torch.zeros(B1, B2)
    for i in range(B1):
        for j in range(B2):
            # transform to (1, 1, 28, 28)
            img1 = rearrange(imgs1[i], "... -> 1 ...")
            img2 = rearrange(imgs2[j], "... -> 1 ...")
            distances[i, j] = ssim_distance(img1, img2)
    return distances


# metrics measures
# Adapted from https://github.com/xuqiantong/
# GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat(
        [torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)], 0
    )
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float("inf")
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(
        k, 0, False
    )

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        "tp": (pred * label).sum(),
        "fp": (pred * (1 - label)).sum(),
        "fn": ((1 - pred) * label).sum(),
        "tn": ((1 - pred) * (1 - label)).sum(),
    }

    s.update(
        {
            "precision": s["tp"] / (s["tp"] + s["fp"] + 1e-10),
            "recall": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_t": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_f": s["tn"] / (s["tn"] + s["fp"] + 1e-10),
            "acc": torch.eq(label, pred).float().mean(),
        }
    )
    return s


def lgan_mmd_cov(distances):
    """
    Compute the Minimum Matching Distance (MMD) between two sets of images.
    """
    N_sample, N_ref = distances.size(0), distances.size(1)
    all_dist = distances
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        "lgan_mmd": mmd,
        "lgan_cov": cov,
        "lgan_mmd_smp": mmd_smp,
    }


def compute_all_metrics(generated, reference):
    result = {}

    distances = pairwise_ssim_distance(generated, reference)
    distances_gen = pairwise_ssim_distance(generated, generated)
    distances_ref = pairwise_ssim_distance(reference, reference)

    # compute knn
    knn_result = knn(distances_gen, distances, distances_ref, 1)
    for key, value in knn_result.items():
        result[f"KNN-{key}"] = value

    # compute lgan_mmd_cov
    result.update(lgan_mmd_cov(distances))
    return result
