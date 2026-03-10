import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def linear_CKA(X, Y):
    """
    X: [N, D]
    Y: [N, D]
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    hsic = torch.norm(Y.T @ X, p='fro') ** 2
    var1 = torch.norm(X.T @ X, p='fro')
    var2 = torch.norm(Y.T @ Y, p='fro')

    return (hsic / (var1 * var2 + 1e-8)).item()


def pairwise_distance_corr(X, Y):
    """
    Correlation between pairwise distance matrices
    """
    Dx = torch.cdist(X, X)
    Dy = torch.cdist(Y, Y)

    Dx = Dx.flatten()
    Dy = Dy.flatten()

    corr = torch.corrcoef(torch.stack([Dx, Dy]))[0,1]
    return corr.item()


def cosine_similarity_tokens(X, Y):
    """
    Mean cosine similarity between corresponding tokens
    X: [N, D]
    Y: [N, D]
    """
    X = F.normalize(X, dim=1)
    Y = F.normalize(Y, dim=1)

    cos = (X * Y).sum(dim=1)
    return cos.mean().item()


def flatten_tokens(tokens):
    """
    tokens: [B, V, T, C] or [B, T, C]
    convert -> [N, C]
    """
    if tokens.dim() == 4:
        B,V,T,C = tokens.shape
        return tokens.reshape(B*V*T, C)
    elif tokens.dim() == 3:
        B,T,C = tokens.shape
        return tokens.reshape(B*T, C)
    else:
        raise ValueError("Unexpected token shape")


def compute_layerwise_metrics(clean_tokens, lq_tokens, res_tokens, layers):

    cka_clean_res = []
    cka_clean_lq = []

    dist_clean_res = []
    dist_clean_lq = []

    cos_clean_res = []
    cos_clean_lq = []

    for l in layers:

        clean = flatten_tokens(clean_tokens[l]).float()
        lq = flatten_tokens(lq_tokens[l]).float()
        res = flatten_tokens(res_tokens[l]).float()

        max_samples = 5000
        if clean.shape[0] > max_samples:
            idx = torch.randperm(clean.shape[0])[:max_samples]
            clean = clean[idx]
            lq = lq[idx]
            res = res[idx]

        # CKA
        cka_clean_res.append(linear_CKA(clean, res))
        cka_clean_lq.append(linear_CKA(clean, lq))

        # distance correlation
        dist_clean_res.append(pairwise_distance_corr(clean, res))
        dist_clean_lq.append(pairwise_distance_corr(clean, lq))

        # cosine similarity
        cos_clean_res.append(cosine_similarity_tokens(clean, res))
        cos_clean_lq.append(cosine_similarity_tokens(clean, lq))

    return {
        "cka_clean_res": cka_clean_res,
        "cka_clean_lq": cka_clean_lq,
        "dist_clean_res": dist_clean_res,
        "dist_clean_lq": dist_clean_lq,
        "cos_clean_res": cos_clean_res,
        "cos_clean_lq": cos_clean_lq
    }

def save_metric_plot(values1, values2, layers, title, save_path, label1="clean-res", label2="clean-lq"):

    plt.figure(figsize=(8,5))

    plt.plot(layers, values1, marker='o', label=label1)
    plt.plot(layers, values2, marker='o', label=label2)

    plt.xlabel("Layer")
    plt.ylabel("Similarity")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()