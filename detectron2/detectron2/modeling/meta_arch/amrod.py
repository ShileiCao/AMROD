# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.optim as optim
from detectron2.config import configurable
from detectron2.layers import move_device_like
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from copy import deepcopy
import math
from .build import META_ARCH_REGISTRY

from collections import OrderedDict
import torch.nn.functional as F
from .losses import  AMRODConLoss
from torchvision.ops import roi_align



__all__ = ["AMROD"]


@META_ARCH_REGISTRY.register()
class AMROD(nn.Module):
    
    @configurable
    def __init__(
        self,
        *,
        model: nn.Module = None,
        model_teacher: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        datasetName: str,
        cfg,
    ):
        super().__init__()
        self.model = model
        self.model_teacher = model_teacher
        self.optimizer = optimizer
        self.steps = 1
        self.model_state_anchor = deepcopy(self.model.state_dict())
        
        self.model_teacher.eval()
        self.model.train()

        self.iter = 0
        self.AMROD_contrastive_loss = AMRODConLoss(temperature=0.07)
        
        self.threshold_init = cfg.SOLVER.THRESHOLD_INIT
        self.mt = cfg.SOLVER.MT
        self.rst_m = cfg.SOLVER.RST_M
        self.loss_weight = cfg.SOLVER.LOSS_WEIGHT
        self.thresholds_max = cfg.SOLVER.THRESHOLD_MAX
        self.thresholds_mini = cfg.SOLVER.THRESHOLD_MINI
        self.alpha_dt = cfg.SOLVER.ALPHA_DT
        self.gamma_dt = cfg.SOLVER.GAMMA_DT
        self.proposals = cfg.SOLVER.PROPOSALS
        
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.thresholds = [self.threshold_init] * self.num_classes
        self.threshold = 0.9
        dim_in = 1024
        
        self.query_head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, dim_in)
            )
        self.optimizer.add_param_group({"params": self.query_head.parameters()})
        
        self.value_head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, dim_in)
            )
        self.optimizer.add_param_group({"params": self.value_head.parameters()})
        
        self.mean_score = [[] for i in range(self.num_classes)]
        self.thresh = [[] for i in range(self.num_classes)]
        self.last_mean = [0.5 for i in range(self.num_classes)]
        
        self.score_window = []
        self.score_em = cfg.SOLVER.SCORE_EM
        self.score_gamma = cfg.SOLVER.SCORE_GAMMA
        self.score_thresh = cfg.SOLVER.SCORE_THRESH
        self.slope_list = []
        self.stop_count = 0
        if datasetName == "ACDC":
            self.totalIter = 400
        else:
            self.totalIter = 500
            

        
        
    @classmethod
    def from_config(cls, cfg):
        model = META_ARCH_REGISTRY.get("GeneralizedRCNN")(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True
        )
        model.to(torch.device(cfg.MODEL.DEVICE))
        
        model_teacher = META_ARCH_REGISTRY.get("GeneralizedRCNN")(cfg)
        DetectionCheckpointer(model_teacher, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True
        )
        model_teacher.to(torch.device(cfg.MODEL.DEVICE))
        for param in model_teacher.parameters():
            param.detach_()
        
        optimizer = build_optimizer(cfg, model)
        
        if cfg.DATASETS.TEST[0] == "c_fog"or cfg.DATASETS.TEST[0] =="fog":
            datasetName = "C"
        elif cfg.DATASETS.TEST[0] == "gaussian_noise":
            datasetName = "C_all"
        elif cfg.DATASETS.TEST[0] == "defocus_blur":
            datasetName = "C_12"
        else:
            datasetName = "ACDC"
        
        return {
            "model": model,
            "model_teacher": model_teacher,
            "optimizer": optimizer,
            "datasetName": datasetName,
            "cfg": cfg
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)
    
    def KD_loss(self, student_logits, teacher_logits) :
        teacher_prob = F.softmax(teacher_logits, dim=1)
        student_log_prob = F.log_softmax(student_logits, dim=1)
        KD_loss = F.kl_div(student_log_prob, teacher_prob.detach(), reduction='batchmean')

        return KD_loss
    
    def dynamic_threshold(self, logits_means):
        new_thresholds = [self.gamma_dt * threshold + (1 - self.gamma_dt) * self.alpha_dt * math.sqrt(mean)
                            if mean>0 else threshold for threshold, mean in zip(self.thresholds, logits_means)]
        new_thresholds = [max(min(threshold, self.thresholds_max), self.thresholds_mini) for threshold in new_thresholds]
        return new_thresholds
    
    def forward(self, x: List[Dict[str, torch.Tensor]]):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, batched_inputs):
        self.iter += 1
        t_features, t_proposals, t_results_raw, outputs = self.model_teacher.inference(batched_inputs, mode= "ctaod")
        

        valid_map = t_results_raw[0].scores > 0.1
        valid_score = t_results_raw[0].scores[valid_map]

        if valid_map.sum() == 0:
            self.stop_count += 1
            return outputs
        else:
            mean_all = valid_score.mean().cpu()
            if self.iter % 50 == 0:
                print("iter: ", self.iter, "score_em", self.score_em)
            if (mean_all / self.score_em) > self.score_thresh or (self.score_em / mean_all) > self.score_thresh:
                self.stop_count += 1
                self.score_em = self.score_gamma * self.score_em + (1-self.score_gamma) * mean_all
                return outputs
            self.score_em = self.score_gamma * self.score_em + (1-self.score_gamma) * mean_all
                
        score_mean = [torch.zeros(1, dtype=t_results_raw[0].scores.dtype, device=t_results_raw[0].scores.device) for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            index = t_results_raw[0].pred_classes[valid_map] == i
            if index.sum() > 0:
                score_mean[i] = valid_score[index].mean()
        
        self.thresholds = self.dynamic_threshold(score_mean)
        t_results = process_pseudo_label(t_results_raw, self.thresholds, True)

            
        loss_dict = {}       
        images = self.model.preprocess_image(batched_inputs, strong_aug = True)
        features = self.model.backbone(images.tensor)
        proposals, proposal_losses = self.model.proposal_generator(images, features, t_results)
        _, detector_losses = self.model.roi_heads(images, features, proposals, t_results)
        
        loss_dict.update(detector_losses)
        loss_dict.update(proposal_losses)
        
        t_proposals[0].proposal_boxes.tensor = t_proposals[0].proposal_boxes.tensor[:self.proposals]
                
        features = [features[f] for f in self.model.roi_heads.box_in_features]
        s_box_features = self.model.roi_heads.box_pooler(features, [x.proposal_boxes for x in t_proposals])
        s_box_features = self.model.roi_heads.box_head(s_box_features)
        s_roih_logits = self.model.roi_heads.box_predictor(s_box_features)
        
        t_features = [t_features[f] for f in self.model_teacher.roi_heads.box_in_features]
        t_box_features = self.model_teacher.roi_heads.box_pooler(t_features, [x.proposal_boxes for x in t_proposals])
        t_box_features = self.model_teacher.roi_heads.box_head(t_box_features)
        t_roih_logits = self.model_teacher.roi_heads.box_predictor(t_box_features)
        

        s_query = self.query_head(s_box_features)
        t_query = self.query_head(t_box_features)
        loss_dict["AMROD"] = self.loss_weight * self.AMROD_contrastive_loss(s_query, t_query)
        
        loss_dict["st_const"] = self.KD_loss(s_roih_logits[0], t_roih_logits[0])  
        
        losses = sum(loss_dict.values())
        assert torch.isfinite(losses).all(), loss_dict        
        self.optimizer.zero_grad()
        losses.backward()
        
        if self.rst_m>0:
            fisher_dict = {}
            for nm, m  in self.model.named_modules():  ## previously used model, but now using self.model
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        fisher_dict[f"{nm}.{npp}"] = p.grad.data.clone().pow(2)
        
        self.optimizer.step()
        
        if self.iter % 50 == 0:
            print("iter: ", self.iter, ''.join(['{0}: {1}, '.format(k, v.item()) for k,v in loss_dict.items()]))
            print("iter: ", self.iter, "thresholds: ", self.thresholds)
            print()
            
        self.model_teacher = update_model(self.model, self.model_teacher, self.mt)
        
        for nm, m  in self.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    data = fisher_dict[f"{nm}.{npp}"]
                    mask = find_weight_quantile(data, self.rst_m)
                    with torch.no_grad(): 
                        p.data = self.model_state_anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)               
            
        return outputs
    
def find_weight_quantile(matrix, perc):    
    weights = matrix / matrix.max()
    weights_with_noise = weights * torch.rand_like(weights)
    
    arr_sorted = torch.sort(weights_with_noise.reshape(-1)).values
    frac_idx = perc*(len(arr_sorted)-1)
    frac_part = frac_idx - int(frac_idx)
    low_idx = int(frac_idx)
    high_idx = low_idx + 1
    threshold = arr_sorted[low_idx] + (arr_sorted[high_idx]-arr_sorted[low_idx]) * frac_part # linear interpolation
    mask = weights_with_noise < threshold
    return mask.float().cuda()


def threshold_bbox(proposal_bbox_inst, thres, dynamic = False):
    
    if dynamic:
        thres = torch.tensor(thres).to(proposal_bbox_inst.scores.device)
        valid_map = torch.gt(proposal_bbox_inst.scores, thres[proposal_bbox_inst.pred_classes])
        
    else:
        valid_map = proposal_bbox_inst.scores > thres

    # create instances containing boxes and gt_classes
    image_shape = proposal_bbox_inst.image_size
    new_proposal_inst = Instances(image_shape)

    # create box
    new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
    new_boxes = Boxes(new_bbox_loc)

    # add boxes to instances
    new_proposal_inst.gt_boxes = new_boxes
    new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
    new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

    return new_proposal_inst


def process_pseudo_label(proposals_rpn_unsup_k, cur_threshold, dynamic = False):
    list_instances = []
    for proposal_bbox_inst in proposals_rpn_unsup_k:
        # thresholding
        proposal_bbox_inst = threshold_bbox(
            proposal_bbox_inst, cur_threshold, dynamic
        )
        list_instances.append(proposal_bbox_inst)
        
    return list_instances


@torch.no_grad()
def update_model(model_student, model_teacher, keep_rate):
    if comm.get_world_size() > 1:
        student_model_dict = {
            key[7:]: value for key, value in model_student.state_dict().items()
        }
    else:
        student_model_dict = model_student.state_dict()

    new_teacher_dict = OrderedDict()
    
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )   
        else:
            raise Exception("{} is not found in student model".format(key))

    model_teacher.load_state_dict(new_teacher_dict)
    return model_teacher

