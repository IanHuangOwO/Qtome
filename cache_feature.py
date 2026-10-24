"""Build the Q-atom-amplitude cache for every finetune dataset of a config (the cache itself: model/Qtome/finetune/feature_cache.py;
train_finetune.py builds it on first use anyway):
    python cache_feature.py --config configs/runs/<backbone>/finetune/<head>/<dataset>_<mode>.json"""
import argparse

from model.Qtome.finetune.feature_cache import get_atom_cache, subjects_of


def main():
    ap = argparse.ArgumentParser(description="Build the atom-amplitude cache for every finetune dataset in a config.")
    ap.add_argument('--config', default='configs/Qtome/finetune.template.json')
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()
    from IO.config import load_config
    config = load_config(args.config)
    for name, ds_args in config['dataset_params']['finetune'].items():
        folder = get_atom_cache(config, name, subjects_of(ds_args), batch_size=args.batch_size)
        print(f"{name}: cache ready in {folder}")


if __name__ == '__main__':
    main()
