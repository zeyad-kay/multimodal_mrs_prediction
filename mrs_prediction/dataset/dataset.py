from glob import glob
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from monai.data.dataloader import DataLoader
from monai.data.image_dataset import ImageDataset
from monai.transforms.compose import Compose
from monai.transforms.utility.array import EnsureChannelFirst
from monai.transforms.croppad.array import CropForeground
from monai.transforms.spatial.array import Spacing, Resize, Orientation
from mrs_prediction.transforms import WindowCT

def split_train_for_cv(train, label_name, cv_splits=5, random_state=1):
    cv = StratifiedKFold(cv_splits, shuffle=True, random_state=random_state)
    splits = []
    for train_ids, val_ids in cv.split(train, train[label_name]):
        splits.append([train.iloc[train_ids], train.iloc[val_ids]])
    return splits


def get_dataloader(df, modality, label_name, transform, batch_size, num_workers, device=""):
    ds = ImageDataset(image_files=df[modality], labels=df[label_name].values, transform=transform, image_only=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory_device=device, pin_memory=True if len(device) else False)
    return dl

def save_data(train, cv_splits, test, statistics, path):
    os.makedirs(path, exist_ok=True)

    train.to_csv(os.path.join(path, "train.csv"), index=False)
    print(f"Training set saved to {os.path.join(path, 'train.csv')}.")

    if test is not None:
        test.to_csv(os.path.join(path, "test.csv"), index=False)
        print(f"Test set saved to {os.path.join(path, 'test.csv')}.")

    for i in range(len(cv_splits)):
        train, val = cv_splits[i]
        os.makedirs(os.path.join(path, f"fold_{i+1}"), exist_ok=True)
        train.to_csv(os.path.join(path, f"fold_{i+1}", "train.csv"), index=False)
        val.to_csv(os.path.join(path, f"fold_{i+1}", "val.csv"), index=False)
        print(f"Fold {i+1} saved to {os.path.join(path, f'fold_{i+1}')}.")

    with open(os.path.join(path, "stats.json") , 'w') as json_file:
        json.dump(statistics, json_file, indent=4)

    print(f"Dataset statistics saved to {os.path.join(path, 'stats.json')}.")

def create_dataset(dataset_dir, modalities, skull_stripped, id_colname, mrs_colname, include_cols=[], exclude_ids=[], join="inner"):
    clinical_data = pd.read_csv(os.path.join(dataset_dir,"clinical.csv"), usecols=[id_colname, mrs_colname, *include_cols])
    clinical_data = clinical_data.dropna()
    clinical_data.loc[clinical_data[mrs_colname] <= 2, "good_outcome"] = 1
    clinical_data.loc[clinical_data[mrs_colname] > 2, "good_outcome"] = 0

    dfs = []

    for modality in modalities:
        subj_with_imgs = [int(Path(subj).stem) for subj in glob(os.path.join(dataset_dir, modality, "*"))]
        subj_with_imgs_and_outcome = clinical_data[(clinical_data[id_colname].isin(subj_with_imgs)) & (~clinical_data[id_colname].isin(np.array(exclude_ids,dtype=int)))]

        ids = []
        paths = []
        for subjid in subj_with_imgs_and_outcome[id_colname]:
            if skull_stripped:
                path = os.path.join(dataset_dir, modality, str(subjid), f"{subjid}_sklstrpd.nii.gz")
            else:
                path = os.path.join(dataset_dir, modality, str(subjid), f"{subjid}.nii.gz")
            if Path(path).exists():
                paths.append(path)
                ids.append(subjid)

        print(f"{modality} has {len(paths)} subjects.")

        if len(paths):
            dfs.append(pd.DataFrame({
                f"{modality}": paths,
            }, index=ids))
        else:
            print(f"Skipping {modality} from join procedure.")

    print(f"Performing {join} join on the dataset...")

    df = (pd.concat(dfs, axis=1, join=join)
            .rename_axis("id")
            .reset_index()
            .merge(clinical_data, left_on="id", right_on=id_colname)
            .sort_values(by=["id"])
            .drop(columns=[id_colname, mrs_colname])
        )

    print(f"Final dataset contains {df.shape[0]} subjects")

    return df

def get_median_spacing(dataloader):
    spacings = []
    for volume, metadata in dataloader:
        print(metadata["filename_or_obj"])
        spacings.append(metadata['pixdim'][0,1:4].tolist())

    return np.median(spacings, axis=0).round(2)

def get_median_shape(dataloader):
    shapes = []
    for volume, metadata in dataloader:
        print(metadata["filename_or_obj"])
        shapes.append(list(volume.shape[2:]))

    return np.median(shapes, axis=0).round(0).astype(int)

def get_intensity_statistics(dataloader):
    num_voxels = 0
    sum_intensity = 0
    sum_intensity = 0
    sum_intensity_squared = 0
    for volume, metadata in dataloader:
        non_zero_voxels = volume[volume > 0]
        num_voxels += non_zero_voxels.shape[0]
        sum_intensity += np.sum(non_zero_voxels)
        sum_intensity_squared += np.sum(non_zero_voxels**2)

    mean = sum_intensity / num_voxels
    std = np.sqrt((sum_intensity_squared + num_voxels*mean**2 - 2*mean*sum_intensity) / num_voxels)
    return np.round(mean, 2).astype(np.float64), np.round(std, 2).astype(np.float64)

def extract_dataset_statistics(train_data):

    # only load modalities columns
    modalities = train_data.columns

    stats = {}

    batch_size = 1

    # train_data = train_data[train_data["B0 CTA"]=="/work/souza_lab/Data/ESCAPE-NA1/B0 CTA/27017/27017_sklstrpd.nii.gz"]
    # print(train_data)
    for modality in modalities:
        stats[modality] = {}

        MIN_THRESHOLD = 0
        # this works in case of MRI
        transform = Compose([EnsureChannelFirst(), Orientation("RAS"), CropForeground(select_fn=lambda x: x > MIN_THRESHOLD, margin=0)])
        if "ncct" in modality.lower():
            # continue
            WINDOW_LEVEL, WINDOW_WIDTH = 40, 80
            MIN_THRESHOLD = WINDOW_LEVEL - WINDOW_WIDTH // 2
            transform = Compose([EnsureChannelFirst(), Orientation("RAS"), WindowCT(WINDOW_LEVEL, WINDOW_WIDTH), CropForeground(select_fn=lambda x: x > MIN_THRESHOLD, margin=0)])
        elif "cta" in modality.lower():
            WINDOW_LEVEL, WINDOW_WIDTH = 250, 400
            MIN_THRESHOLD = WINDOW_LEVEL - WINDOW_WIDTH // 2
            transform = Compose([EnsureChannelFirst(), Orientation("RAS"), WindowCT(WINDOW_LEVEL, WINDOW_WIDTH), CropForeground(select_fn=lambda x: x > MIN_THRESHOLD, margin=0)])

        img_pths = train_data[modality].values

        ds = ImageDataset(image_files=img_pths, labels=None, transform=transform, image_only=False)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=3)

        print(f"Extracting dataset statistics for {modality}")
        print(f"Extracting median spacing and shape...")

        median_spacing = get_median_spacing(dl)
        median_shape = get_median_shape(dl)

        stats[modality]["target_spacing"] = median_spacing.tolist()
        stats[modality]["cropped_median_shape"] = median_shape.tolist()

        print(f"Median spacing: {median_spacing}")
        print(f"Median shape after foreground cropping: {median_shape}")

        # this works in case of MRI
        transform = Compose([EnsureChannelFirst(), Orientation("RAS"), CropForeground(select_fn=lambda x: x > MIN_THRESHOLD, margin=0), Spacing(median_spacing)])
        if "ncct" in modality.lower():
            WINDOW_LEVEL, WINDOW_WIDTH = 40, 80
            MIN_THRESHOLD = WINDOW_LEVEL - WINDOW_WIDTH // 2
            transform = Compose([EnsureChannelFirst(), Orientation("RAS"), WindowCT(WINDOW_LEVEL, WINDOW_WIDTH), CropForeground(select_fn=lambda x: x > MIN_THRESHOLD, margin=0), Spacing(median_spacing)])
        elif "cta" in modality.lower():
            WINDOW_LEVEL, WINDOW_WIDTH = 250, 400
            MIN_THRESHOLD = WINDOW_LEVEL - WINDOW_WIDTH // 2
            transform = Compose([EnsureChannelFirst(), Orientation("RAS"), WindowCT(WINDOW_LEVEL, WINDOW_WIDTH), CropForeground(select_fn=lambda x: x > MIN_THRESHOLD, margin=0), Spacing(median_spacing)])

        ds = ImageDataset(image_files=img_pths, labels=None, transform=transform, image_only=False)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=3)

        print("Extracting median shape after resampling to target spacing...")

        median_shape = get_median_shape(dl)
        stats[modality]["target_shape"] = median_shape.tolist()

        print(f"Median shape after resampling: {median_shape}")

        if "ncct" in modality.lower():
            WINDOW_LEVEL, WINDOW_WIDTH = 40, 80
            MIN_THRESHOLD = WINDOW_LEVEL - WINDOW_WIDTH // 2
            transform = Compose([EnsureChannelFirst(), Orientation("RAS"), WindowCT(WINDOW_LEVEL, WINDOW_WIDTH), CropForeground(select_fn=lambda x: x > 0, margin=0), Spacing(median_spacing), Resize(median_shape)])
        if "cta" in modality.lower():
            WINDOW_LEVEL, WINDOW_WIDTH = 250, 400
            MIN_THRESHOLD = WINDOW_LEVEL - WINDOW_WIDTH // 2
            transform = Compose([EnsureChannelFirst(), Orientation("RAS"), WindowCT(WINDOW_LEVEL, WINDOW_WIDTH), CropForeground(select_fn=lambda x: x > 0, margin=0), Spacing(median_spacing), Resize(median_shape)])

            ds = ImageDataset(image_files=img_pths, labels=None, transform=transform, image_only=False)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=3)

            print("Extracting intensity statistics after resizing to target shape...")

        mean_intensity, std_intensity = get_intensity_statistics(dl)
        stats[modality]["target_mean_intensity"] = mean_intensity
        stats[modality]["target_std_intensity"] = std_intensity

        print(f"Mean intensity: {mean_intensity}")
        print(f"std intensity: {std_intensity}")

    return stats
