set -ex 
### Setting GPU
export CUDA_VISIBLE_DEVICES=0

### setting benchmark and data path
benchmark=charades-sta # or activitynet-captions
feature_path=features
annotation_path=annotations

if [ "$benchmark" = "charades-sta" ]; then
    vid_feature_path="${feature_path}/charades_i3d_features"
elif [ "$benchmark" = "activitynet-captions" ]; then
    vid_feature_path="${feature_path}/activitynet_c3d_features"
else
    echo "Error: benchmark ${benchmark} is not supported!"
    exit 1
fi

### setting running parameters
v_feat_dim=1024 # 500 for activitynet-captions
num_vid_clips=72
num_enc_layers=4
num_dec_layers=4
num_epoch=200 # for activitynet, setting 'num_epoch' to 100 is enough 

### running training and eval. For more detailed parameter configurations, please refer to src/config.py
python src/train_val.py \
    --dataset ${benchmark} \
    --vid_feature_path ${vid_feature_path} \
    --annotation_path ${annotation_path} \
    --v_feat_dim ${v_feat_dim} \
    --num_clips ${num_vid_clips} \
    --enc_layers ${num_enc_layers} \
    --dec_layers ${num_dec_layers} \
    --n_epoch ${num_epoch} 