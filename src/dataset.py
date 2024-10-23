""" Dataset loader for the Charades-STA dataset """
import os
import numpy as np
import torch
import torch.utils.data as data
from datasets import load_dataset
from utils.tensor_utils import pad_sequences_1d

class BaseDataset(data.Dataset):
    def __init__(self, split, opt):
        super(BaseDataset, self).__init__()
        self.args = opt
        self.dataset = opt.dataset
        self.data_dir = os.path.join('annotations', self.dataset)
        self.split = split
        self.load_labels = True
        self.num_clips = opt.num_clips
        self.annotations = load_dataset("json", data_files=os.path.join(self.data_dir, '{}.jsonl'.format(split)), split="train").to_list()[:1000]

    def __getitem__(self, index):
        meta = dict()
        model_inputs = dict()
        qid = self.annotations[index]['qid']
        vid = self.annotations[index]['vid']
        description = self.annotations[index]['query']
        duration = self.annotations[index]['duration']
        gt_period = self.annotations[index]['relevant_windows'][0]

        gt_period_tensor = torch.tensor(gt_period).div(duration)
        center = gt_period_tensor.sum(-1) * 0.5
        width = gt_period_tensor[..., 1] - gt_period_tensor[..., 0]
        gt_period_tensor = torch.stack([center, width], dim=-1).unsqueeze(0)

        # RoBERTa or CLIP version
        visual_input =  torch.from_numpy(np.load(os.path.join(self.args.vid_feature_path, f"{vid}.npy"))).float()
        visual_input = self.average_to_fixed_length(visual_input)

        meta['qid'] = qid
        meta['query'] = description
        meta['duration'] = duration
        meta['vid'] = vid
        meta['relevant_windows'] = gt_period
        model_inputs['video_feat'] = visual_input
        model_inputs['query'] = description
        model_inputs['span_labels'] = gt_period_tensor
        ans = dict(meta=meta, model_inputs=model_inputs)
        return ans

    def __len__(self):
        return len(self.annotations)
    
    def average_to_fixed_length(self, visual_input):
        num_clips = visual_input.shape[0]
        idxs = torch.arange(0, self.num_clips+1, 1.0)/self.num_clips*num_clips
        idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1))
        new_visual_input = []
        for i in range(self.num_clips):
            s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
            if s_idx < e_idx:
                new_visual_input.append(torch.mean(visual_input[s_idx:e_idx],dim=0))
            else:
                new_visual_input.append(visual_input[s_idx])
        new_visual_input = torch.stack(new_visual_input, dim=0)
        return new_visual_input


def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        if k in ["query"]:
            batched_data[k] = [e["model_inputs"][k] for e in batch]
            continue
        batched_data[k] = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
        query=batched_model_inputs["query"],
    )

    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]

    targets = None if len(targets) == 0 else targets
    return model_inputs, targets
