import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="charades-sta", choices=["charades-sta", "activitynet-captions"])
    parser.add_argument("--eval_split_name", type=str, default="val", help="should match keys in video_duration_idx_path, must set for VCMR")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--seed", type=int, default=2018, help="random seed")
    parser.add_argument("--num_workers", type=int, default=12, help="num subprocesses used to load the data, 0: use main process")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="choosing devices for training")
    
    # Training config
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="lr ratio at the beginning")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epochs to run")
    parser.add_argument("--max_es_cnt", type=int, default=15, help="number of evaluations to early stop, use -1 to disable early stop")
    parser.add_argument("--train_bsz", type=int, default=64, help="mini-batch size")
    parser.add_argument("--eval_bsz", type=int, default=128, help="mini-batch size at inference, for query")
    parser.add_argument("--grad_clip", type=float, default=0.1, help="perform gradient clip, -1: disable")
    parser.add_argument("--eval_epoch_interval", type=int, default=5, help="evaluation interval epoch during training")

    # Data config
    parser.add_argument("--max_q_l", type=int, default=32)
    parser.add_argument("--max_v_l", type=int, default=128)
    parser.add_argument("--num_clips", type=int, default=64, help='video sequence length')
    parser.add_argument("--clip_length", type=float, default=0.5)
    parser.add_argument("--max_windows", type=int, default=10)
    parser.add_argument("--v_feat_dim", type=int, default=1024 , help="video feature dim")
    parser.add_argument("--vid_feature_path", type=str, help='Path to the video features') 
    parser.add_argument("--annotation_path", type=str, default="./annotations", help="which feature are you using for charades")
    
    # Diffusion config
    parser.add_argument("--num_timesteps", type=int, default=1000 , help="diffusion time steps")
    parser.add_argument("--sampling_timesteps", type=int, default=1 , help="ddim sampling steps")
    parser.add_argument("--ddim_sampling_eta", type=int, default=1 , help="ddim sampling eta")
    parser.add_argument("--scale", type=float, default=2.0 , help="the scale to strengthen the spans signal")

    # Transformer config
    parser.add_argument('--enc_layers', default=4, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--input_dropout', default=0.5, type=float, help="Dropout applied in input")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=5, type=int, help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true') 
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument("--span_loss_type", default="l1", type=str, choices=['l1', 'ce'], help="l1: (center-x, width) regression. ce: (st_idx, ed_idx) classification.")
    parser.add_argument('--span_loss_coef', default=3, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)

    # Post processing
    parser.add_argument("--no_sort_results", action="store_true", help="do not sort results, use this for moment query visualization")
    parser.add_argument("--max_before_nms", type=int, default=10)
    parser.add_argument("--max_after_nms", type=int, default=10)
    parser.add_argument("--conf_thd", type=float, default=0.0, help="only keep windows with conf >= conf_thd")
    parser.add_argument("--nms_thd", type=float, default=-1, help="additionally use non-maximum suppression (or non-minimum suppression for distance) to post-processing the predictions. -1: do not use nms. [0, 1]")
    
    return parser.parse_args()