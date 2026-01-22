import torch
import torch.nn as nn
import numpy as np

def loss_weight_dict(labels: np.ndarray, num_b: int):
    counts, bin_edges = np.histogram(labels, bins=num_b)

    weight_dict = {}
    for i, count in enumerate(counts):
        if count > 0:  # 0인 bin 제거
            if np.max(labels)==bin_edges[i+1]:
                bin_range = (bin_edges[i], bin_edges[i + 1]+np.float64(1e-5))
            else:
                bin_range = (bin_edges[i], bin_edges[i + 1])
            weight_dict[bin_range] = count

    if len(weight_dict) != num_b:
        print("Total number of boundaries created:", len(weight_dict))
        print("Be careful with the k setting")

    return weight_dict

def compute_labelwise_feature_mean(features: torch.Tensor, labels: torch.Tensor, group_list):
    if labels.ndim == 2:
        labels = labels.squeeze(-1)

    unique_labels = labels.unique(sorted=True)
    feature_means = []

    for group in group_list:
        mask = torch.zeros(len(labels), device=labels.device, dtype=torch.bool)

        for low, high in group:
            mask |= (labels >= low) & (labels < high)

        selected = features[mask]

        if selected.ndim == 1:
            selected = selected.unsqueeze(0)

        if selected.shape[0] == 0:
            mean = torch.zeros(features.shape[1], device=features.device)
        else:
            mean = selected.mean(dim=0)

        feature_means.append(mean)

    feature_means = torch.stack(feature_means, dim=0)

    return feature_means, unique_labels


def l2_error_per_feature(labels, features, mean_feature, group_list, group_weight):
    if labels.dim() == 2:
        labels = labels.squeeze()

    l2_list = []
    i=0
    for group in group_list:
        mask = torch.zeros(len(labels), device=labels.device, dtype=torch.bool)

        for low, high in group:
            mask |= (labels >= low) & (labels < high)
        count = mask.sum()

        if count <= 1:
            continue

        diffs = features[mask] - mean_feature[i]
        squared = (diffs ** 2).sum(dim=1)

        l2 = squared.mean()
        l2_list.append(l2*torch.exp(torch.tensor(group_weight[i])))
        i+=1
    return l2_list


def group_dict_keys(data_dict: dict, n: int) -> list:
    if len(data_dict) % n != 0:
        raise ValueError("The boundaries should be divisible by k.")

    keys = list(data_dict.keys())
    grouped = [keys[i:i + n] for i in range(0, len(keys), n)]
    return grouped

def weight_by_clusture(data_dict, group_list):
    group_sums = []

    for group in group_list:
        group_sum = sum(data_dict.get(key, 0) for key in group)
        group_sums.append(group_sum)

    max_value = max(group_sums)

    diffs = [max_value - v for v in group_sums]

    max_diff = max(diffs) if max(diffs) != 0 else 1
    normalized = [d / max_diff for d in diffs]

    new_dict = {i: norm for i, norm in enumerate(normalized)}
    return new_dict

class CenterIR(nn.Module):
    def __init__(self, weight=None, k=None):
        super(CenterIR, self).__init__()
        self.weight = weight
        self.k = k

    def forward(self, features, labels):

        loss = torch.zeros(1, device=features.device)
        for group_size in self.k:
            group_list = group_dict_keys(self.weight, group_size)

            group_weight = weight_by_clusture(self.weight, group_list)

            mean_feature, labels_list = compute_labelwise_feature_mean(features, labels, group_list)

            local_loss = l2_error_per_feature(labels, features, mean_feature, group_list, group_weight)

            loss+=(torch.stack(local_loss)).mean()

        return loss