from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn

from torch.utils.data import DataLoader, Dataset
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


class IndexedDataset(Dataset):
    """
    Wrap a base dataset + an index list, while preserving attributes
    like max_seq_len, feature_df, class_names needed by Exp_Classification.
    """
    def __init__(self, base_ds, indices):
        self.base = base_ds
        self.indices = np.asarray(indices, dtype=np.int64)

        # expose metadata used elsewhere
        for attr in ["max_seq_len", "feature_df", "class_names", "feature_names"]:
            if hasattr(base_ds, attr):
                setattr(self, attr, getattr(base_ds, attr))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[int(self.indices[i])]


def _uea_labels_1d(ds):
    # labels_df is created from categorical codes (needed for CE loss) :contentReference[oaicite:4]{index=4}
    y = ds.labels_df.values
    y = np.squeeze(y)
    if y.ndim > 1:
        y = y[:, 0]
    return y.astype(int)


def _get_cached_uea_objects(args):
    if not hasattr(args, "_uea_cache"):
        args._uea_cache = {}
    return args._uea_cache


def _get_stratified_indices(args, y):
    """
    Returns (train_idx, val_idx) for:
      - single stratified split if args.k_fold <= 1
      - stratified k-fold if args.k_fold > 1 (val fold = args.fold)
    """
    seed = getattr(args, "seed", 2021)
    val_ratio = getattr(args, "val_ratio", 0.2)
    k_fold = int(getattr(args, "k_fold", 0) or 0)
    fold = int(getattr(args, "fold", 0) or 0)

    cache = _get_cached_uea_objects(args)
    key = ("split", len(y), seed, val_ratio, k_fold, fold)
    if key in cache:
        return cache[key]

    if k_fold and k_fold > 1:
        if fold < 0 or fold >= k_fold:
            raise ValueError(f"--fold must be in [0, {k_fold-1}], got {fold}")

        # Note: each class should have at least k_fold samples for “perfect” stratification
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed)
        splits = list(skf.split(np.zeros(len(y)), y))
        train_idx, val_idx = splits[fold]
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(sss.split(np.zeros(len(y)), y))

    cache[key] = (train_idx, val_idx)
    return train_idx, val_idx


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    flag_u = str(flag).upper()
    shuffle_flag = False if (flag_u == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        data_set = Data(
            args=args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader

    elif args.task_name == 'classification':
        # Special handling for UEA: create VAL from TRAIN (stratified), keep TEST separate
        if args.data == 'UEA':
            cache = _get_cached_uea_objects(args)

            if flag_u in ('TRAIN', 'VAL'):
                # Build (and cache) the full TRAIN dataset twice:
                # - TRAIN version: augmentation may run when flag == "TRAIN" :contentReference[oaicite:5]{index=5}
                # - VAL version: no augmentation
                if "uea_train_full" not in cache:
                    cache["uea_train_full"] = Data(args=args, root_path=args.root_path, flag='TRAIN')
                if "uea_val_full" not in cache:
                    cache["uea_val_full"] = Data(args=args, root_path=args.root_path, flag='VAL')

                y = _uea_labels_1d(cache["uea_train_full"])
                train_idx, val_idx = _get_stratified_indices(args, y)

                if flag_u == 'TRAIN':
                    data_set = IndexedDataset(cache["uea_train_full"], train_idx)
                    shuffle_flag = True
                else:
                    data_set = IndexedDataset(cache["uea_val_full"], val_idx)
                    shuffle_flag = False

            else:
                # TEST stays the real *_TEST.ts
                data_set = Data(args=args, root_path=args.root_path, flag='TEST')
                shuffle_flag = False

        else:
            # Non-UEA classification (unchanged)
            data_set = Data(
                args=args,
                root_path=args.root_path,
                flag=flag,
            )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader

    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader
