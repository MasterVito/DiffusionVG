# torch==2.0.1, transformers==4.29.1
import torch
import torch.nn.functional as F
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel

import copy
import math

# The main part of our proposed DiffusionVG model
class DiffusionVG(nn.Module):
    def __init__(self, encoder, decoder, position_embed,  txt_position_embed):
        super().__init__()
        # Diffusion Parameters
        self.num_timesteps = 1000
        self.sampling_timesteps = 5 
        self.ddim_sampling_eta = 1.0
        self.scale = 2.0
        self.num_queries = 5
        betas = cosine_beta_schedule(timesteps=self.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.) 
        # pre-calculated diffusion parameters 
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_log_variance_clipped', torch.log(betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod).clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.encoder = encoder
        self.decoder = decoder
        self.vid_position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        self.hidden_dim = encoder.d_model
        self.span_embed = MLP(self.hidden_dim, self.hidden_dim, 2, 3) # project span into high dimension
        self.input_txt_proj = LinearLayer(768, self.hidden_dim, layer_norm=True, dropout=0.1, relu=True)  # sentence feature dimension reduction
        self.input_vid_proj = LinearLayer(1024, self.hidden_dim, layer_norm=True, dropout=0.1, relu=True) # video feature dimension reduction
        self.aux_loss = True
        
    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, span_labels = None):
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        pos_vid = self.vid_position_embed(src_vid, src_vid_mask)
        pos_txt = self.txt_position_embed(src_txt)
        pos_embed, mask, memory = self.encoder(vid_src = src_vid, txt_src = src_txt, vid_mask = ~src_vid_mask.bool(), txt_mask = src_txt_mask, vid_pos = pos_vid, txt_pos = pos_txt)
        
        # inference stage with DDIM sampling
        output = self.ddim_sample(mask, pos_embed, memory)
        return output
    
    # get predicted noise with predicted x_0 and x_t
    def predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    # predict the target span at time step t
    def model_predictions(self, mask, pos_embed, memory, x, t):
        x_spans = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_spans = ((x_spans / self.scale) + 1) / 2
        x_spans = span_cxw_to_xx(x_spans) 
        outputs_coord = self.decoder(mask, pos_embed, memory, x_spans, t, None)
        x_start = outputs_coord[-1]  
        x_start = span_xx_to_cxw(x_start) 
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start, outputs_coord, x_spans

    def ddim_sample(self, mask, pos_embed, memory):
        batch = memory.shape[1]
        shape = (batch, self.num_queries, 2)
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  
        
        targets = torch.randn(shape, device=memory.device) 
        targets = (torch.clamp(targets, min=-1, max=1) + 1) / 2
        targets = span_cxw_to_xx(targets)
        
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=memory.device, dtype=torch.long)
            pred_noise, x_start, outputs_coord, cur_spans = self.model_predictions(mask, pos_embed, memory, targets, time_cond)
            # Finish sampling
            if time_next < 0:
                targets = x_start
                continue
            # DDIM_step
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(targets)
            targets = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        output = {'pred_spans': span_xx_to_cxw(outputs_coord[-1])}
        return output


# span refining decoder
class SpanRefiningDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 256
        self.dim_feedforward = 1024
        self.nhead = 8
        self.dropout = 0.1
        self.activation = 'relu'
        self.num_layers = 2
        self.feature_scorer = nn.Linear(self.d_model, 1)
        denoise_layer = SpanRefiningDecoderLayer(self.activation)
        self.denoise_series = _get_clones(denoise_layer, self.num_layers)
        self.return_intermediate = True
        
        # Gaussian random feature embedding layer for time
        time_dim = self.d_model * 4
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(self.d_model), nn.Linear(self.d_model, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
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
        num_boxes = spans.shape[1]
        
        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            query_features = init_features.clone()
        else:
            query_features = None
                
        for layer_idx, denoise_layer in enumerate(self.denoise_series):
            pred_bboxes, query_features = denoise_layer(mask = mask, pos_embed = pos_embed, memory = memory, spans = spans, query_features = query_features, time_emb = time, scorer = self.feature_scorer)
            if self.return_intermediate:
                inter_pred_bboxes.append(pred_bboxes)
            spans = pred_bboxes.detach()
    
        if self.return_intermediate:
            return torch.stack(inter_pred_bboxes)
        return pred_bboxes[None] 


# span refining decoder layer
class SpanRefiningDecoderLayer(nn.Module):
    def __init__(self, activation="relu",bbox_weights=(2.0, 1.0)): 
        super().__init__()
        self.d_model = 256
        # decoder layer components
        self.cross_attn = nn.MultiheadAttention(self.d_model, 8, dropout=0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(self.d_model)
        
        self.linear1 = nn.Linear(self.d_model, self.d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.activation = _get_activation_fn(activation)
        # block time mlp
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(self.d_model * 4, self.d_model * 2))
        self.vid_length = 64
        
        self.reg_module = MLP(self.d_model, self.d_model, 2, 3)
        self.bbox_weights = bbox_weights
        self.span_embed = MLP(2, self.d_model, self.d_model, 2)
            
    # crop the feature segments corresponds to given noisy spans generate span features with a weighted sum operation
    def span_feature_align(self, memory, spans, scorer = None):
        N, nr_boxes = spans.shape[:2]
        proposal_spans = torch.clamp((spans * self.vid_length).type(torch.int), min = 0, max = 63)
        vis_memory = memory[:self.vid_length].permute(1, 0, 2)
        span_features = []
        for bsz_id in range(N):
            cur_batch_span_features = []
            for span_id in range(nr_boxes):
                if proposal_spans[bsz_id, span_id, 0] == proposal_spans[bsz_id, span_id, 1]:
                    span_feature = vis_memory[bsz_id, proposal_spans[bsz_id, span_id, 0].item()].unsqueeze(0)
                else:
                    span_feature = vis_memory[bsz_id, proposal_spans[bsz_id, span_id, 0]:proposal_spans[bsz_id, span_id, 1]]
                span_feature_scores = F.softmax(scorer(span_feature), dim=0)
                span_feature = torch.sum(span_feature_scores * span_feature, dim=0)
                cur_batch_span_features.append(span_feature)
            cur_batch_span_features = torch.stack(cur_batch_span_features)
            span_features.append(cur_batch_span_features)
        span_features = torch.stack(span_features)
        return span_features
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
        
    def forward(self, mask, pos_embed, memory, spans, query_features, time_emb, scorer):
        N, nr_boxes = spans.shape[:2]
        
        span_features = self.span_feature_align(memory = memory, spans = spans, scorer = scorer) # get the mean-pooling features from memory
        span_embedding = self.span_embed(spans.to(torch.float32))
        span_features = span_features + span_embedding

        # first layer there is no updated span features
        if query_features is None:
            query_features = span_features.clone()
        
        span_features = span_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)
        
        # Query Feature updating.
        query_features = query_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.cross_attn(query = query_features, key = span_features, value = span_features)[0]
        query_features = query_features + self.dropout2(pro_features2) 
        obj_features = self.norm2(query_features)
        
        # Feed forward layer.
        obj_features2 = self.linear1(self.dropout1(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout1(obj_features2)
        obj_features = self.norm1(obj_features)
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)

        # Time embedding
        scale_shift = self.block_time_mlp(time_emb) 
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0) 
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift

        reg_feature = fc_feature.clone()
        spans_deltas = self.reg_module(reg_feature) # estimate the deviations between input spans and ground-truth
        pred_spans = self.apply_deltas(spans_deltas, spans.view(-1, 2))
        return pred_spans.view(N, nr_boxes, -1), obj_features

    # Update current noisy spans with the estimate deviations
    def apply_deltas(self, deltas, spans):
        spans = spans.to(deltas.dtype)
        widths = spans[:, 1::2] - spans[:, 0::2] 
        center = spans[:, 0::2] + 0.5 * widths
        wc, ww = self.bbox_weights
        dc = deltas[:, 0::2] / wc
        dw = deltas[:, 1::2] / ww
        dw = torch.clamp(dw, max=5)         
        pred_center = center + dc               
        pred_width = torch.sigmoid(dw + widths)  
        return span_cxw_to_xx(torch.cat((pred_center, pred_width), dim=-1))


# video-centered multi-modal encoder
class VideoCenteredEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4, dim_feedforward=1024, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        encoder_layer = VideoCenteredEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        self.norm = nn.LayerNorm(d_model) if normalize_before else None
        self.num_layers = num_encoder_layers
        self.layers = _get_clones(encoder_layer, self.num_layers)
        self.d_model = d_model
        self.nhead = nhead
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, vid_src, txt_src, vid_mask, txt_mask, vid_pos, txt_pos):
        vid_src = vid_src.permute(1, 0, 2)
        txt_src = txt_src.permute(1, 0, 2)
        vid_pos = vid_pos.permute(1, 0, 2) 
        txt_pos = txt_pos.permute(1, 0, 2)  
        
        for layer in self.layers:
            vid_src = layer(vid_src, txt_src, vid_mask, txt_mask, vid_pos, txt_pos)
        if self.norm is not None:
            vid_src = self.norm(vid_src)
        return torch.cat([vid_pos, txt_pos], dim = 0), torch.cat([vid_mask, txt_mask], dim = 1), torch.cat([vid_src, txt_src], dim = 0)
  

# video-centered multi-modal encoder layer
class VideoCenteredEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, activation="relu", normalize_before=False, text_in_self_attention = False):
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

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.text_in_self_attention = text_in_self_attention

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, vid_src, txt_src, vid_mask, txt_mask, vid_pos, txt_pos):
        # video self-attention
        q = k = self.with_pos_embed(vid_src, vid_pos)
        tgt2 = self.self_attn(q, k, value=vid_src, key_padding_mask=vid_mask)[0]
        vid_src = vid_src + self.dropout1(tgt2)
        vid_src = self.norm1(vid_src)
        
        # text-to-video self-attention
        tgt2 = self.multihead_attn(query=self.with_pos_embed(vid_src, vid_pos), key=self.with_pos_embed(txt_src, txt_pos), value=txt_src, attn_mask=None, key_padding_mask=txt_mask)[0]
        vid_src = vid_src + self.dropout2(tgt2)
        vid_src = self.norm2(vid_src)
        
        # video FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(vid_src))))
        vid_src = vid_src + self.dropout3(tgt2)
        vid_src = self.norm3(vid_src)
        return vid_src


# pre-trained Distil-BERT model
class DistilBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.body = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for p in self.body.parameters(): p.requires_grad_(False)

    def forward(self, texts, device ='cpu'):
        tokenized = self.tokenizer.batch_encode_plus(texts, padding="longest", return_tensors="pt").to(device)
        encoded_text = self.body(**tokenized)
        text_memory = encoded_text.last_hidden_state.transpose(0, 1)
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        return text_attention_mask, text_memory.permute(1,0,2)


# sine positional embediing for video features
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
    
    
# trainable positional embediing for sentence features
class TrainablePositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# positional embedding for time step t
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        assert mask is not None
        x_embed = mask.cumsum(1, dtype=torch.float32)  # (bsz, L)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2)) / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t  # (bsz, L, num_pos_feats)
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2) 
        return pos_x  
    
    
# simple multi-layer perceptron (also called FFN)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# linear layer configurable with layer normalization, dropout, ReLU.
class LinearLayer(nn.Module):
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm: self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.layer_norm: x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu: x = F.relu(x, inplace=True)
        return x  # (N, L, D)


# turn the format of the span to [center, width] 
def span_xx_to_cxw(xx_spans):
    center = xx_spans.sum(-1) * 0.5
    width = xx_spans[..., 1] - xx_spans[..., 0]
    return torch.stack([center, width], dim=-1)

# turn the format of the span to [start, end] 
def span_cxw_to_xx(cxw_spans):
    x1 = cxw_spans[..., 0] - 0.5 * cxw_spans[..., 1]
    x2 = cxw_spans[..., 0] + 0.5 * cxw_spans[..., 1]
    return torch.stack([x1, x2], dim=-1)

# create the monotonical decreasing cosine noise schedule
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

# extract the appropriate  t  index for a batch of indices
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# Return an activation function given a string
def _get_activation_fn(activation):
    if activation == "relu": return F.relu
    if activation == "gelu": return F.gelu
    if activation == "glu":  return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# build positional embedding for video and sentence features
def build_position_encoding():
    vid_position_embedding = PositionEmbeddingSine(256, normalize=True)
    txt_position_embedding = TrainablePositionalEncoding(max_position_embeddings=32, hidden_size=256, dropout=0.1)
    return vid_position_embedding, txt_position_embedding

# Build our DiffusionVG model
def build_model():
    encoder = VideoCenteredEncoder()
    decoder = SpanRefiningDecoder()
    vid_position_embedding, txt_position_embedding = build_position_encoding()
    model = DiffusionVG(encoder, decoder, vid_position_embedding, txt_position_embedding)
    return model


if __name__=='__main__':
    text_encoder = DistilBERT()
    model = build_model().eval()                                          # sampling stage
    sentence_query = ['A man is running in the playground.']              # query sentence
    sentence_mask, sentence_features = text_encoder(sentence_query)       # use pretrained distilbert-base to extract sentence features
    video_features = torch.randn(size = (1, 64, 1024))                    # input video features
    video_mask = torch.ones(size = (1, 64))                               # video feature mask
    gaussian_noise_inputs = torch.randn(size=(1,2))                       # Gaussian noise inputs
    prediction = model(sentence_features, sentence_mask, video_features, video_mask, gaussian_noise_inputs)
    # prediction['pred_spans'] are the final predictions in our DiffusionVG (without voting process), the formats of the predicted spans are [center, width]