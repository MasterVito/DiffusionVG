import os
import time
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_scheduler

from src.config import parse_args
from src.dataset import BaseDataset
from src.model import build_model
from src.post_process import PostProcessorDETR
from src.span_utils import span_cxw_to_xx
from src.dataset import start_end_collate, prepare_batch_inputs
from src.eval import eval_submission

from utils.basic_utils import AverageMeter, dict_to_markdown, save_json, save_jsonl, save_json_full
from utils.model_utils import count_parameters
from utils.temporal_nms import temporal_nms, calculate_IoU_batch


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        
def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, lr_scheduler):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)
    timer_dataloading = time.time()
    
    for batch_idx, batch in tqdm(enumerate(train_loader), total = len(train_loader)):
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)
        timer_start = time.time()
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
        time_meters["prepare_inputs_time"].update(time.time() - timer_start)
        timer_start = time.time()

        model_inputs.update(targets)
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        time_meters["model_forward_time"].update(time.time() - timer_start)
        timer_start = time.time() 
        optimizer.zero_grad()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        lr_scheduler.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))
        timer_dataloading = time.time()

    logger.info("Epoch time stats:")
    for name, meter in time_meters.items():
        d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
        logger.info(f"{name} ==> {d}")


def train(model, criterion, optimizer, lr_scheduler, opt):
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    train_loader = DataLoader(opt.train_dataset, collate_fn=start_end_collate, batch_size=opt.train_bsz, num_workers=opt.num_workers, shuffle=True)
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dataset, opt.eval_split_name)

    for epoch_i in range(opt.n_epoch):
        train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, lr_scheduler)
        prev_best_score = 0.0
        # evaluation
        if (epoch_i + 1) % opt.eval_epoch_interval == 0:
            logger.info(f'Evaluating {epoch_i + 1}th epoch')
            val_metrics_no_nms, val_metrics_nms = eval_epoch(model, opt.test_dataset, opt, save_submission_filename, epoch_i, criterion)
            print("validation metrics_no_nms {}".format(pprint.pformat(val_metrics_no_nms["brief"], indent=4)))
            if val_metrics_nms is not None:
                print("metrics_nms {}".format(pprint.pformat(val_metrics_nms["brief"], indent=4)))
            eval_score = val_metrics_no_nms["brief"]["MR-full-R1@0.7"]
            if eval_score > prev_best_score:
                prev_best_score = eval_score
                logger.info(f"New best R1@0.3: {val_metrics_no_nms['brief']['MR-full-R1@0.3']}, R1@0.5: {val_metrics_no_nms['brief']['MR-full-R1@0.5']}, R1@0.7: {val_metrics_no_nms['brief']['MR-full-R1@0.7']}")


def post_processing_mr_nms(mr_res, nms_thd, max_before_nms, max_after_nms):
    mr_res_after_nms = []
    for e in mr_res:
        e["pred_relevant_windows"] = temporal_nms(
            e["pred_relevant_windows"][:max_before_nms],
            nms_thd=nms_thd,
            max_after_nms=max_after_nms
        )
        mr_res_after_nms.append(e)
    return mr_res_after_nms


def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename, epoch_i):
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_root, save_submission_filename)
    # evaluation
    metrics = eval_submission(submission, gt_data, verbose=False, match_number=True)
    save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
    save_json_full(metrics, epoch_i, save_metrics_path, save_pretty=False, sort_keys=False)
    latest_file_paths = [submission_path, save_metrics_path]

    if opt.nms_thd != -1:
        logger.info("[MR] Performing nms with nms_thd {}".format(opt.nms_thd))
        submission_after_nms = post_processing_mr_nms(
            submission, nms_thd=opt.nms_thd,
            max_before_nms=opt.max_before_nms, max_after_nms=opt.max_after_nms
        )

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(".jsonl", "_nms_thd_{}.jsonl".format(opt.nms_thd))
        save_jsonl(submission_after_nms, submission_nms_path)
        if opt.eval_split_name == "val":
            metrics_nms = eval_submission(
                submission_after_nms, gt_data,
                verbose=opt.debug, match_number=not opt.debug
            )
            save_metrics_nms_path = submission_nms_path.replace(".jsonl", "_metrics.json")
            save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [submission_nms_path, ]
    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths


@torch.no_grad()
def compute_mr_results(model, eval_loader, opt, criterion=None):
    model.eval()
    criterion.eval()

    loss_meters = defaultdict(AverageMeter)

    with torch.no_grad():
        mr_res = []
        for batch in tqdm(eval_loader, total = len(eval_loader), desc="Evaluating..."):
            query_meta = batch[0]
            model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
            model_inputs.update(targets)
            outputs = model(**model_inputs)
            
            ### Voting for the final answer span
            bsz = len(batch[0])            
            num_props = opt.num_queries
            scores = torch.ones(len(batch[0]), num_props)  # * (batch_size, #queries)  foreground label is 0, we directly take it
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, 2)
            pred_spans = span_cxw_to_xx(pred_spans)
            numpy_spans = pred_spans.cpu().numpy()

            if opt.dataset == 'charades-sta':
                # On Charades-STA, the IoU of many proposals is small, and it doesn't make sense to get these proposals to vote. 
                # So we weight the voting results of each proposal according to it's IoU with the first proposal.
                c = np.zeros((bsz, num_props))
                for i in range(num_props):
                    iou = calculate_IoU_batch((numpy_spans[:, 0, 0], numpy_spans[:, 0, 1]), (numpy_spans[:, i, 0], numpy_spans[:, i, 1]))
                    c[:, i] = iou
            else:
                c = np.ones((bsz, num_props))
            scores = np.zeros((bsz, num_props))
            for i in range(num_props):
                for j in range(num_props):
                    iou = calculate_IoU_batch((numpy_spans[:, i, 0], numpy_spans[:, i, 1]), (numpy_spans[:, j, 0], numpy_spans[:, j, 1]))
                    iou = iou * c[:, j]
                    scores[:, i] = scores[:, i] + iou
            scores = torch.from_numpy(scores)
                
            ### compose predictions
            for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans.cpu(), scores.cpu())):
                spans = spans * meta["duration"]
                cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
                if not opt.no_sort_results:
                    cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
                cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
                cur_query_pred = dict(
                    qid=meta["qid"],
                    query=meta["query"],
                    vid=meta["vid"],
                    pred_relevant_windows=cur_ranked_preds,
                )
                mr_res.append(cur_query_pred)

            if criterion:
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                loss_dict["loss_overall"] = float(losses)  # for logging only
                for k, v in loss_dict.items():
                    loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

    max_ts_val = {"charades-sta": 120, "activitynet-captions": 275}
    post_processor = PostProcessorDETR(clip_length=opt.clip_length, min_ts_val=0, max_ts_val=max_ts_val[opt.dataset], min_w_l=2, max_w_l=150, move_window_method="left", process_func_names=("clip_ts", "round_multiple"))
    mr_res = post_processor(mr_res)
    return mr_res, loss_meters


def eval_epoch(model, eval_dataset, opt, save_submission_filename, epoch_i=None, criterion=None):
    logger.info("Generate submissions")
    model.eval()
    criterion.eval()

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False)

    submission, _ = compute_mr_results(model, eval_loader, opt, criterion)
    if opt.no_sort_results:
        save_submission_filename = save_submission_filename.replace(".jsonl", "_unsorted.jsonl")
    metrics, metrics_nms, _ = eval_epoch_post_processing(submission, opt, eval_dataset.annotations, save_submission_filename, epoch_i)
    return metrics, metrics_nms


def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")
    model, criterion = build_model(opt)
    count_parameters(model)
    model.to("cuda")
    criterion.to("cuda")

    ### lower the learning rate of text encoder to 10% of         
    txt_lr = opt.lr * 0.1
    text_enc_param = [p for n, p in model.named_parameters() if (("text_encoder" in n) and p.requires_grad)]
    rest_param = [p for n, p in model.named_parameters() if(('vis_encoder' not in n) and ('text_encoder' not in n) and p.requires_grad)]
    param_dicts = [{"params" : rest_param}, {"params" : text_enc_param, "lr" : txt_lr}]
        
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    num_steps = (len(opt.train_dataset) // opt.train_bsz) * opt.n_epoch
    lr_scheduler = get_scheduler(name="cosine", optimizer=optimizer, num_training_steps=num_steps, num_warmup_steps=int(num_steps * opt.warmup_ratio))
    return model, criterion, optimizer, lr_scheduler


def start_training():
    logger.info("Setup config, data and model...")
    set_seed(opt.seed)
    opt.train_dataset = BaseDataset('train', opt)
    opt.test_dataset = BaseDataset('test', opt)
    
    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info("Start Training...")
    train(model, criterion, optimizer, lr_scheduler, opt)
    return opt.ckpt_filepath, opt.eval_split_name


if __name__ == '__main__':
    opt = parse_args()
    start_training()
