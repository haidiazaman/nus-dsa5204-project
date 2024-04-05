import os
from monai.data import CacheDataset, load_decathlon_datalist


def get_btcv_dataset(path, data_json_name, subset="training", transform=None):
    # training, validation, testing
    data_json_path = os.path.join(path, data_json_name)
    datalist = load_decathlon_datalist(data_json_path, True, subset)
    dataset = CacheDataset(
        data=datalist,
        transform=transform,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    return dataset
