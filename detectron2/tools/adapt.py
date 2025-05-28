#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from detectron2.utils.visualizer import Visualizer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from contextlib import ExitStack
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.data.datasets import register_pascal_voc, register_ACDC_instances, register_cityscape
import cv2
import pandas as pd



logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, allow_cached_coco=False))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

def adaptAndEval(model, data_loader, evaluator):
    evaluator.reset()
    with EventStorage() as storage:
        for iter, data in enumerate(data_loader):
            outputs = model(data)
            evaluator.process(data, outputs)
        return evaluator.evaluate()
    


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    data_loaders = [build_detection_test_loader(cfg, datasetName) for datasetName in cfg.DATASETS.TEST]
    evaluators = [get_evaluator(cfg, datasetName, os.path.join(cfg.OUTPUT_DIR, "inference", datasetName)) for datasetName in cfg.DATASETS.TEST]
    
    AP50_Sum = 0
    count = 0
    results = []
    epochs = 10
    
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True
        )
        epochs = 1
        model.eval()
    
    
    if cfg.DATASETS.TEST[0] == "c_fog" or cfg.DATASETS.TEST[0] == "fog":
        datasetName = "C"
    elif cfg.DATASETS.TEST[0] == "defocus_blur":
        datasetName = "C_12"
        epochs = 1
    elif cfg.DATASETS.TEST[0] =="shift_cloudy":
        datasetName = "shift"
        epochs = 1
    else:
        datasetName = "ACDC"
        
    
    fileName = f"{cfg.MODEL.META_ARCHITECTURE}(-{datasetName} em-{cfg.SOLVER.SCORE_EM} gamma-{cfg.SOLVER.SCORE_GAMMA} thresh-{cfg.SOLVER.SCORE_THRESH} mt-{cfg.SOLVER.MT} init-{cfg.SOLVER.THRESHOLD_INIT} mini-{cfg.SOLVER.THRESHOLD_MINI} alpha-{cfg.SOLVER.ALPHA_DT} gamma{cfg.SOLVER.GAMMA_DT} restore-{cfg.SOLVER.RST_M} weight-{cfg.SOLVER.LOSS_WEIGHT} proposal-{cfg.SOLVER.PROPOSALS} AMRODcon).xlsx"


    for epoch in range(epochs):
        one_round_result = OrderedDict()
        for iter, data_loader, evaluator in zip(range(len(data_loaders)), data_loaders, evaluators):
            print("epoch:", epoch,", dataSet:", iter)
            evaluator.reset()
            result = adaptAndEval(model, data_loader, evaluator)
            one_round_result[cfg.DATASETS.TEST[iter]] = result
            AP50_Sum += result['bbox']['AP50']
            count += 1
            print("Current AP mean(",fileName,"):", round(AP50_Sum/count, 1))
        results.append(one_round_result)
        
    AP50_ALL = []
    for epoch, one_round_result in enumerate(results):
        for dataset_name in cfg.DATASETS.TEST:
            AP50 = one_round_result[dataset_name]['bbox']['AP50']
            AP50_ALL.append(round(AP50,1))
            print("round:", epoch+1, "dataset:", dataset_name, "AP50:", AP50)
            
    AP50_ALL_mean = round(sum(AP50_ALL)/len(AP50_ALL), 1)
    AP50_ALL.append(AP50_ALL_mean)
    print("AP50_ALL_mean:", AP50_ALL_mean)
    
    df = pd.DataFrame([AP50_ALL])
    df.to_excel(fileName, index=False, header=False)
    print("save to excel file:", fileName)
    print("stop_count:", model.stop_count)
    
    

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)   
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
