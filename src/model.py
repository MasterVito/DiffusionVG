import math
import copy
import random
import numpy as np
from time import time
from pdb import set_trace
from typing import Optional
from collections import namedtuple
from easydict import EasyDict as edict
from transformers import DistilBertTokenizer, DistilBertModel

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from src.span_utils import generalized_temporal_iou, span_cxw_to_xx, span_xx_to_cxw
from src.position_encoding import build_position_encoding
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        
        
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Parameters for setting cosine decreasing betas for forward diffusion
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def extract(a, t, x_shape):
    """
    extract the appropriate  t  index for a batch of indices
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    

class DiffusionVG(nn.Module):
    """ 
    The proposed DiffusionVG model, including multimodal encoding, noise generation in forward diffusion, 
    and denoising modules in reverse diffusion. 
    """
    def __init__(self, 
                 encoder,
                 head, 
                 position_embed, 
                 txt_position_embed, 
                 args):
        super().__init__()
        self.num_timesteps = args.num_timesteps
        betas = cosine_beta_schedule(timesteps=self.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.) 
        self.sampling_timesteps = args.sampling_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
        self.ddim_sampling_eta = args.ddim_sampling_eta
        self.self_condition = False
        self.scale = args.scale
        self.span_renewal = True
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        self.num_queries = args.num_queries
        self.encoder = encoder
        self.head = head
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = encoder.d_model
        
        self.span_loss_type = args.span_loss_type
        self.max_v_l = args.max_v_l
        span_pred_dim = 2 if self.span_loss_type == "l1" else args.max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        
        self.input_vid_proj = LinearLayer(args.v_feat_dim, hidden_dim, layer_norm=True, dropout=args.input_dropout, relu=True)
        self.aux_loss = args.aux_loss
        self.text_encoder = TextEncoder(outdim=hidden_dim, device=args.device, freeze = False)

    def forward(self, query, src_vid, src_vid_mask, span_labels):
        src_vid = self.input_vid_proj(src_vid)
        src_txt_mask, src_txt, src_cls = self.text_encoder(query)
        src_txt_mask = ~src_txt_mask
        src_txt = src_txt.permute(1,0,2)
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt)
        pos_embed, mask, memory, attn = self.encoder(vid_src = src_vid, txt_src = src_txt, vid_mask = ~src_vid_mask.bool(), txt_mask = ~src_txt_mask.bool(), vid_pos = pos_vid, txt_pos = pos_txt)

        if self.training:
            spans = torch.cat([item['spans'] for item in span_labels], dim = 0)
            x_spans, noises, t = self.prepare_targets(spans) # targerts: cw
            t = t.squeeze(-1)
            outputs_coord = self.head(mask, pos_embed, memory, x_spans, t, None) # diffusiondet output: outputs_class[lyr, bsz, num_q, cls] outputs_coord[lyr, bsz, num_q, 4]
            outputs_coord = span_xx_to_cxw(outputs_coord) # head的output是xx格式的
            output = {'pred_spans': outputs_coord[-1]}

            if self.aux_loss: 
                output['aux_outputs'] = [{'pred_spans': b} for b in outputs_coord[:-1]]
            return output
        
        if not self.training:
            output = self.ddim_sample(mask, pos_embed, memory)
            output['attention'] = attn
            return output
        
    def predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def model_predictions(self, mask, pos_embed, memory, x, t):
        x_spans = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_spans = ((x_spans / self.scale) + 1) / 2
        x_spans = span_cxw_to_xx(x_spans)
        outputs_coord = self.head(mask, pos_embed, memory, x_spans, t, None)
        
        x_start = outputs_coord[-1] 
        x_start = span_xx_to_cxw(x_start) 
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start), outputs_coord, x_spans

    def ddim_sample(self, mask, pos_embed, memory):
        batch = memory.shape[1]
        shape = (batch, self.num_queries, 2)
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        inter_noisy_spans = []
        targets = torch.randn(shape, device=memory.device)
        targets = (torch.clamp(targets, min=-1, max=1) + 1) / 2
        targets = span_cxw_to_xx(targets)
        inter_noisy_spans.append(targets)
        
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=memory.device, dtype=torch.long)
            preds, outputs_coord, cur_spans = self.model_predictions(mask, pos_embed, memory, targets, time_cond)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            if time_next > 0: 
                inter_noisy_spans.append(cur_spans)
            if time_next < 0:
                targets = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(targets)
            targets = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise # 预测的下一个采样值：x_(tau-1)
            
        output = {'pred_spans': span_xx_to_cxw(outputs_coord[-1]), 'all_spans': torch.stack(inter_noisy_spans)}
        return output

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def prepare_targets(self, spans):
        bsz = spans.shape[0]
        ts = torch.randint(0, self.num_timesteps, (bsz,1)).to(spans.device).long()    
        noises = torch.randn(bsz, self.num_queries, 2).to(spans.device)               
        start_spans = spans.unsqueeze(1).repeat(1, self.num_queries, 1)
        start_spans = (start_spans * 2. - 1.) * self.scale                              
        noised_spans = torch.stack([self.q_sample(x_start=start_spans[i], t=ts[i], noise=noises[i]) for i in range(bsz)])   
        noised_spans = torch.clamp(noised_spans, min=-1 * self.scale, max=self.scale)   
        noised_spans = ((noised_spans / self.scale) + 1) / 2.                          
        noised_spans_xx = span_cxw_to_xx(noised_spans)
        return noised_spans_xx, noises, ts


class SpanRefiningDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.hidden_dim
        self.dim_feedforward = args.dim_feedforward
        self.nhead = args.nheads
        self.dropout = args.dropout
        self.num_layers = args.dec_layers
        self.feature_scorer = nn.Linear(self.d_model, 1)
        denoise_layer = SpanRefiningDecoderLayer(args)
        self.denoise_series = nn.ModuleList([copy.deepcopy(denoise_layer) for i in range(self.num_layers)])
        self.return_intermediate = args.aux_loss 
        time_dim = self.d_model * 4
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(self.d_model), 
                                      nn.Linear(self.d_model, time_dim), 
                                      nn.GELU(), 
                                      nn.Linear(time_dim, time_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, mask, pos_embed, memory, x_spans, t, init_features = None): 
        time = self.time_mlp(t)
        inter_pred_bboxes = []
        bs = len(memory[0])
        spans = x_spans
        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None
                
        for layer_idx, denoise_layer in enumerate(self.denoise_series):
            pred_bboxes, proposal_features = denoise_layer(mask = mask, pos_embed = pos_embed, memory = memory, spans = spans, pro_features = proposal_features, time_emb = time, scorer = self.feature_scorer)
            if self.return_intermediate:
                inter_pred_bboxes.append(pred_bboxes)
            spans = pred_bboxes.detach()
    
        if self.return_intermediate:
            return torch.stack(inter_pred_bboxes)
        return pred_bboxes[None]


class SpanRefiningDecoderLayer(nn.Module):
    """
    Decoder layer for denoising in DDIM
    """
    def __init__(self, args, bbox_weights=(2.0, 1.0)):
        super().__init__()
        self.d_model = args.hidden_dim
        # dynamic.
        self.cross_attn = nn.MultiheadAttention(args.hidden_dim, args.nheads, dropout=args.dropout)
        self.linear1 = nn.Linear(args.hidden_dim, args.dim_feedforward)
        self.linear2 = nn.Linear(args.dim_feedforward, args.hidden_dim)

        self.norm2 = nn.LayerNorm(args.hidden_dim)
        self.norm3 = nn.LayerNorm(args.hidden_dim)
        
        self.dropout = nn.Dropout(args.dropout)
        self.dropout2 = nn.Dropout(args.dropout)
        self.dropout3 = nn.Dropout(args.dropout)

        # block time mlp
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(args.hidden_dim * 4, args.hidden_dim * 2))
        self.vid_length = args.num_clips
        
        self.span_embed = MLP(2, args.hidden_dim, args.hidden_dim, 2)
        self.reg_module = MLP(args.hidden_dim, args.hidden_dim, 2, 3)
        self.bbox_weights = bbox_weights

    def get_rot_features(self, memory, spans, scorer = None):
        """
        Obtaining span level feature 
        """
        N, nr_boxes = spans.shape[:2]
        proposal_spans = torch.clamp((spans * self.vid_length).type(torch.int), min = 0, max = self.vid_length-1)
        vis_memory = memory[:self.vid_length].permute(1, 0, 2)
        span_features = []
        for bsz_id in range(N):
            cur_batch_span_features = []
            for span_id in range(nr_boxes):
                if proposal_spans[bsz_id, span_id, 0] == proposal_spans[bsz_id, span_id, 1]:
                    span_feature = vis_memory[bsz_id, proposal_spans[bsz_id, span_id, 0].item()].unsqueeze(0)
                else:
                    span_feature = vis_memory[bsz_id, proposal_spans[bsz_id, span_id, 0]:proposal_spans[bsz_id, span_id, 1]] # 会出现负的
                span_feature_scores = F.softmax(scorer(span_feature), dim=0)
                span_feature = torch.sum(span_feature_scores * span_feature, dim=0)
                
                cur_batch_span_features.append(span_feature)
            cur_batch_span_features = torch.stack(cur_batch_span_features)
            span_features.append(cur_batch_span_features)
        span_features = torch.stack(span_features)
        return span_features
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, mask, pos_embed, memory, spans, pro_features, time_emb, scorer = None):
        N, nr_boxes = spans.shape[:2]
        dim = memory.shape[-1]
        span_features = self.get_rot_features(memory = memory, spans = spans, scorer = scorer)

        span_embedding = self.span_embed(spans.to(torch.float32))
        span_features = span_features + span_embedding

        if pro_features is None:
            pro_features = span_features.clone()
        
        span_features = span_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)

        # cross attention layer
        pro_features2 = self.cross_attn(query = pro_features, key = span_features, value = span_features)[0]
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # feed-forward layer
        obj_features2 = self.linear2(self.dropout(F.relu(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)
        
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        scale_shift = self.block_time_mlp(time_emb) # torch.Size([4, 512])，对time的FFN
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0) # 在dim重复nr_boxes次
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift
        reg_feature = fc_feature.clone()
        spans_deltas = self.reg_module(reg_feature)
        pred_spans = self.apply_deltas(spans_deltas, spans.view(-1, 2))
        return pred_spans.view(N, nr_boxes, -1), obj_features

    def apply_deltas(self, deltas, spans):
        spans = spans.to(deltas.dtype)
        widths = spans[:, 1::2] - spans[:, 0::2]  # 加上这个::2能够保留这一维度
        center = spans[:, 0::2] + 0.5 * widths
        wc, ww = self.bbox_weights
        dc = deltas[:, 0::2] / wc
        dw = deltas[:, 1::2] / ww
        dw = torch.clamp(dw, max=5)              # Prevent sending too large values into torch.exp()
        pred_center = center + dc                # center + △center
        pred_width = torch.sigmoid(dw + widths)  # exp(△width) 必须保证是正的，尝试将dw作为width的偏移量，考虑用sigmoid来计算width
        return span_cxw_to_xx(torch.cat((pred_center, pred_width), dim=-1))


class SetCriterion(nn.Module):
    """
    Calculating span-level loss function
    """
    def __init__(self, args):
        super().__init__()
        self.weight_dict = args.weight_dict
        self.losses = args.losses
        self.span_loss_type = args.span_loss_type
        self.max_v_l = args.max_v_l
    
    def loss_spans(self, outputs, targets):
        assert 'pred_spans' in outputs
        batch, num_proposals = outputs['pred_spans'].shape[:2]
        targets = targets["span_labels"]
        src_spans = outputs['pred_spans'].view(batch * num_proposals, 2) # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'] for t in targets], dim=0).unsqueeze(1).repeat(1, num_proposals, 1).view(batch * num_proposals, 2).to(src_spans.device)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {"spans": self.loss_spans}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))
            
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ 
    Very simple multi-layer perceptron (also called FFN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        torch.manual_seed(2018)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """
    linear layer configurable with layer normalization, dropout, ReLU.
    """
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    

class TextEncoder(nn.Module):
    """
    For encoding text embeddings
    """
    def __init__(self, outdim, device, freeze=False) -> None:
        super().__init__()
        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.body = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc = nn.Linear(self.body.config.dim, outdim, bias=True)
        self.layer_norm = nn.LayerNorm(outdim, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        if freeze:
            for p in self.body.parameters():
                p.requires_grad_(False)

    def forward(self, texts):
        tokenized = self.tokenizer.batch_encode_plus(texts, padding="longest", return_tensors="pt").to(self.device)
        encoded_text = self.body(**tokenized)
        text_memory = encoded_text.last_hidden_state.transpose(0, 1)
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        # projecting text features into the same dimension as video features
        text_memory = self.fc(text_memory)
        text_memory = self.layer_norm(text_memory)
        text_memory = self.dropout(text_memory)
        return text_attention_mask, text_memory, None


class VideoCenteredEncoder(nn.Module):
    """
    Video centered multi-modal encoder
    """
    def __init__(self, args):
        super().__init__()
        encoder_layer = VideoCenteredEncoderLayer(args.hidden_dim, args.nheads, args.dim_feedforward, args.dropout, args.pre_norm)
        self.norm = nn.LayerNorm(args.hidden_dim) if args.pre_norm else None
        self.num_layers = args.enc_layers
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(self.num_layers)])
        self.d_model = args.hidden_dim
        self.nhead = args.nheads
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, vid_src, txt_src, vid_mask, txt_mask, vid_pos, txt_pos):
        vid_src = vid_src.permute(1, 0, 2)  # (L, batch_size, d)
        txt_src = txt_src.permute(1, 0, 2)
        vid_pos = vid_pos.permute(1, 0, 2)   # (L, batch_size, d)
        txt_pos = txt_pos.permute(1, 0, 2)   # (L, batch_size, d)
        
        for layer in self.layers:
            vid_src, attn = layer(vid_src, txt_src, vid_mask, txt_mask, vid_pos, txt_pos)
        if self.norm is not None:
            vid_src = self.norm(vid_src)
        return torch.cat([vid_pos, txt_pos], dim = 0), torch.cat([vid_mask, txt_mask], dim = 1), torch.cat([vid_src, txt_src], dim = 0), attn
  
    
class VideoCenteredEncoderLayer(nn.Module):
    """
    Video centered multi-modal encoding layer, including video self-attention, video-to-text cross attention, and FFN for video
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, vid_src, txt_src, vid_mask, txt_mask, vid_pos, txt_pos):
        # video self-attention
        q = k = self.with_pos_embed(vid_src, vid_pos)
        tgt2 = self.self_attn(q, k, value=vid_src, key_padding_mask=vid_mask)[0]
        vid_src = vid_src + self.dropout1(tgt2)
        vid_src = self.norm1(vid_src)
        # video-text cross-attention
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(vid_src, vid_pos), key=self.with_pos_embed(txt_src, txt_pos), value=txt_src, attn_mask=None, key_padding_mask=txt_mask)
        vid_src = vid_src + self.dropout2(tgt2)
        vid_src = self.norm2(vid_src)
        # video FFN
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(vid_src))))
        vid_src = vid_src + self.dropout3(tgt2)
        vid_src = self.norm3(vid_src)
        return vid_src, attn
    

def build_model(args):
    encoder = VideoCenteredEncoder(args)
    head = SpanRefiningDecoder(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)
    model = DiffusionVG(encoder, head, position_embedding, txt_position_embedding, args)
    args.weight_dict = {"loss_span": args.span_loss_coef, "loss_giou": args.giou_loss_coef}
    
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in args.weight_dict.items() if k != "loss_saliency"})
        args.weight_dict.update(aux_weight_dict)
        
    args.losses = ["spans"]
    criterion = SetCriterion(args)
    return model, criterion

