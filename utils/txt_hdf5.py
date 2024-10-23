def make_hdf5_files(args, savedir):
    hdf5path = f"/apdcephfs_cq3/share_1567347/vitoliang/datasets/tsgv_features/{args.dataset.lower()}_text_roberta_features.hdf5"
    f = h5py.File(hdf5path, "a")
    for saved_features in savedFiles:
        textFeaturePath = os.path.join(savedir, saved_features)
        textFeatureName = saved_features.split('.')[0]
        if textFeatureName not in list(f.keys()):
            textFeature = torch.load(textFeaturePath, map_location='cpu').squeeze(0).detach().numpy().astype(np.float32)
            d = f.create_dataset(textFeatureName, data=textFeature)
    f.close()


if __name__=='__main__':
    args = parse_args()
    update_config(args.cfg)
    args.dataset = config.DATASET.NAME
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = RobertaTokenizerFast.from_pretrained('/apdcephfs_cq3/share_1567347/vitoliang/datasets/model_caches/roberta-base', local_files_only=True)
    text_encoder = RobertaModel.from_pretrained('/apdcephfs_cq3/share_1567347/vitoliang/datasets/model_caches/roberta-base', local_files_only=True).to(device) # type: ignore

    savedir = os.path.join('./textFeatures', args.dataset.lower())
    if not os.path.exists(savedir): os.mkdir(savedir)
    savedFiles = os.listdir(savedir)

    convert_dataset_to_entence_feature(args)
    # make_hdf5_files(args, savedir)
