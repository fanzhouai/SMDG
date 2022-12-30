import torch
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn

def compute_wasserstein(features, btch_sz, feature_extractor, discriminator, use_gp = True, gp_weight = 0.1):

    num_domains = int(len(features)/btch_sz)
    dis_loss = 0
    for t in range(num_domains):

        for k in range(t + 1, num_domains):
            
            features_t = features[t * btch_sz:(t + 1) * btch_sz]
            features_k = features[k * btch_sz:(k + 1) * btch_sz]
            
            dis_t = discriminator(features_t)
            dis_k = discriminator(features_k)
            
            if use_gp:
                gp = gradient_penalty(discriminator, features_t, features_k)
                disc_loss = dis_t.mean() - dis_k.mean() - gp_weight*gp
            else: 
                disc_loss = dis_t.mean() - dis_k.mean()



            dis_loss += disc_loss
    
    return dis_loss

# setting gradient values
def set_requires_grad(model, requires_grad=True):
    """
    Used in training adversarial approach
    :param model:
    :param requires_grad:
    :return:
    """

    for param in model.parameters():
        param.requires_grad = requires_grad


# setting gradient penalty for sure the lipschitiz property
def gradient_penalty(critic, h_s, h_t):
    ''' Gradeitnt penalty for Wasserstein GAN'''
    alpha = torch.rand(h_s.size(0), 1).cuda()
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
    # interpolates.requires_grad_()
    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()

    return gradient_penalty


class MultiSimilarityLoss(nn.Module):
    def __init__(self, cfg):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = cfg['scale_pos']
        self.scale_neg = cfg['scale_neg']

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))
        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss

def sort_loaders(loaders_list, reverse=True):
    num_loaders = len(loaders_list)
    
    tuple_ = []
    
    for i in range(num_loaders):
        tuple_.append((loaders_list[i],len(loaders_list[i])))
    
    return sorted(tuple_, key=lambda tuple_len: tuple_len[1],reverse=reverse)