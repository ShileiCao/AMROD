
from __future__ import print_function

import torch
import torch.nn as nn

import torch.nn.functional as F


class AMRODConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(AMRODConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, s_box_features, t_box_features):
        
        batch_size, _ = s_box_features.shape
        mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        
        anchor_feat = F.normalize(s_box_features, dim=1)
        contrast_feat = F.normalize(t_box_features, dim=1)
        
        logits = torch.div(torch.matmul(anchor_feat, contrast_feat.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        sm_logits = logits - logits_max.detach()
        s_all_logits = torch.exp(sm_logits)
        log_prob = sm_logits - torch.log(s_all_logits.sum(1, keepdim=True))
        loss = -1 * ((mask * log_prob).sum(1) / mask.sum(1))   

        if torch.isnan(loss.mean()):
            loss = loss*0
            
        return loss.mean()
