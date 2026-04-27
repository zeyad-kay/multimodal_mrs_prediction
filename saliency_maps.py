import os
import numpy as np
import pandas as pd
import shap
import torch
from glob import glob
from matplotlib import pyplot as plt
from monai.data.dataloader import DataLoader
from monai.data.image_dataset import ImageDataset
from mrs_prediction.model_zoo import init_model
from mrs_prediction.transforms import get_test_transforms, get_train_transforms
from mrs_prediction.utils import load_config, parse_shap_args

def set_inplace_relu_to_false(module, name):
    
    if type(module) in [torch.nn.ReLU, torch.nn.LeakyReLU]:
        setattr(module, 'inplace', False)

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) in [torch.nn.ReLU, torch.nn.LeakyReLU]:
            setattr(module, attr_str, type(target_attr)(inplace=False))

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        set_inplace_relu_to_false(immediate_child_module, name)

def sigmoid_wrapper(module, input, output):
    return torch.nn.Sigmoid()(output)

def prepare_model(model_name, model_checkpoint):
    model = init_model(model_name, out_channels=1)

    checkpoint = torch.load(model_checkpoint, weights_only=True, map_location="cpu")
    checkpoint['model_state_dict'] = {k.replace("module.",""):v for k,v in checkpoint['model_state_dict'].items()}
    checkpoint['model_state_dict'] = {k.replace("_orig_mod.",""):v for k,v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(checkpoint['model_state_dict'])

    set_inplace_relu_to_false(model, model_name)

    model.classifier.register_forward_hook(sigmoid_wrapper)

    return model

def inverse_intensity_normalization(test_volume, mean_intensity, std_intensity):
    return (test_volume * std_intensity) + mean_intensity

def save_shap_maps(path, test_volume, shap_values, delta):
    
    os.makedirs(path,exist_ok=True)
    
    np.save(f"{os.path.join(path, 'shap')}.npy", shap_values)

    with open(f"{os.path.join(path, 'log')}", "w") as f:
        f.write(f"F(x) - E(X) - Sum(phi) = {delta}")

    # manipulate dimensions to treat each slice as an image
    test_volume = torch.transpose(test_volume, 2, 3).transpose(4,1).squeeze(0).numpy()
    shap_values = np.transpose(shap_values, (0,4,3,2,1,5)).squeeze(0).squeeze(-1)
    
    vmax = np.max(np.abs(shap_values))*1e-1
    
    for i in range(test_volume.shape[0]):
        
        slice_shap_values = np.expand_dims(shap_values[i], 0)
        slice_test_volume = np.expand_dims(test_volume[i], 0)

        plt.figure(dpi=600)
        
        # flip to show images in RAS orientation
        shap.plots.image(np.flip(np.fliplr(slice_shap_values),(2)), np.flip(np.fliplr(slice_test_volume),(2)), vmax=vmax, show=False)

        plt.savefig(f"{os.path.join(path, str(i))}.png")
        
        plt.close("all")

def check_shap_additivity(explainer, model, input, shap_values):
    with torch.no_grad():
        model_output_values = model(input)

    assert len(explainer.expected_value) == model_output_values.shape[1], (
        "Length of expected values and model outputs does not match."
    )

    for t in range(len(explainer.expected_value)):
        if not explainer.explainer.multi_input:
            diffs = (
                model_output_values[:, t]
                - explainer.expected_value[t]
                - shap_values[t].sum(axis=tuple(range(1, shap_values[t].ndim)))
            )
        else:
            diffs = model_output_values[:, t] - explainer.expected_value[t]

            for i in range(len(shap_values[t])):
                diffs -= shap_values[t][i].sum(axis=tuple(range(1, shap_values[t][i].ndim)))

        maxdiff = np.abs(diffs).max()

        return maxdiff


if __name__ == "__main__":

    # run_name = "run-20260418_161918-uffngnd4"
    # bg_samples = 2
    # viz_window_level = 40
    # viz_window_width = 80

    args = parse_shap_args()
    run_name = args.run
    bg_samples = args.bg_samples
    viz_window_level = args.wl
    viz_window_width = args.ww

    samples_of_interest = [
        5027,
        5048,
        5052,
        7017,
        7032,
        10011,
        23002,
        24017,
        25008,
        28019,
        28033,
        28066,
        28074,
        40001,
        40006,
        40010,
        41005,
        42005,
        42013,
        73001,
        6004,
        12005,
        12037,
        13046,
        20009,
        22006,
        31030,
        31054,
        31094,
        37015,
        38002,
        38005,
        38014,
        39001,
        39003,
        43003,
        51019,
        60008,
        70009,
        70013,
        77006
    ]


    # fltrd = [5027, 5048, 5052, 6004, 7017, 7032, 10011, 12005, 12037, 13046, 20009, 22006, 23002, 24017, 25008, 28019, 28033, 28066, 28074, 31030, 31054, 31094, 37015, 38002, 38005, 38007, 38014, 39001, 39003, 43003, 51019, 60008, 70009, 70013, 77006]
    # fltrd = [5027,5048,5052,7017,7032,10011,23002,24017,25008,28019,28033,28066,28074,40001,40006,40010,41005,42005,42013,73001,6004,12005,12037,13046,20009,22006,31030,31054]

    # samples_of_interest = list(filter(lambda x: x not in fltrd, samples_of_interest))

    output_dir = os.path.join("saliency_maps", f"{run_name}_shap")

    run_config = os.path.join("wandb", run_name, "files", "config.yaml")

    cfg = load_config(run_config)

    data_root_dir = cfg["data"]["value"]["path"]
    train_fold_path = os.path.join(data_root_dir, "fold_0", "train.csv")

    model_name = cfg["model"]["value"]["name"]
    model_checkpoint = os.path.join("wandb", run_name, "files", f"{model_name}.tar")

    targets = [t["target"] for t in cfg["tasks"]["value"]]
    modality = cfg["data"]["value"]["modality"]

    preproc_params = cfg["preprocessing"]["value"]

    voxel_spacing = preproc_params["spacing"]
    image_size = preproc_params["size"]
    window_level, window_width = preproc_params["windowing"]["level"], preproc_params["windowing"]["width"]
    mean_intensity, std_intensity = preproc_params["normalize"]["mean"], preproc_params["normalize"]["std"]

    viz_window_level = viz_window_level or window_level
    viz_window_width = viz_window_width or window_width

    model = prepare_model(model_name, model_checkpoint)

    # bg_transform = get_train_transforms(modality, window_level=window_level, window_width=window_width, voxel_spacing=voxel_spacing, image_size=image_size, mean_intensity=mean_intensity, std_intensity=std_intensity)
    bg_transform = get_test_transforms(modality, window_level=window_level, window_width=window_width, voxel_spacing=voxel_spacing, image_size=image_size, mean_intensity=mean_intensity, std_intensity=std_intensity)
    samples_of_interest_transform = get_test_transforms(modality, window_level=window_level, window_width=window_width, voxel_spacing=voxel_spacing, image_size=image_size, mean_intensity=mean_intensity, std_intensity=std_intensity)
    viz_samples_of_interest_transform = get_test_transforms(modality, window_level=viz_window_level, window_width=viz_window_width, voxel_spacing=voxel_spacing, image_size=image_size, mean_intensity=0, std_intensity=1)

    bg_df = pd.read_csv(train_fold_path).sample(frac=1.0, random_state=1)

    class0 = bg_df[bg_df[targets[0]] == 0].sample(n=bg_samples//2, random_state=1)
    class1 = bg_df[bg_df[targets[0]] == 1].sample(n=bg_samples//2, random_state=1)

    bg_df = pd.concat([class0, class1]).sample(frac=1.0, random_state=1)

    all_samples = pd.concat([pd.read_csv(pth) for pth in glob(os.path.join(data_root_dir, "*.csv"))])

    samples_of_interest_df = all_samples[all_samples["id"].isin(samples_of_interest)].sort_values(by="id")

    bg_ds = ImageDataset(image_files=bg_df[modality].values, labels=bg_df[targets].values, transform=bg_transform, image_only=False) # type: ignore
    bg_loader = DataLoader(bg_ds, batch_size=bg_samples, shuffle=False, num_workers=0)

    samples_ds = ImageDataset(image_files=samples_of_interest_df[modality].values, labels=samples_of_interest_df[targets].values, transform=samples_of_interest_transform, image_only=False) # type: ignore
    samples_loader = DataLoader(samples_ds, batch_size=1, shuffle=False, num_workers=1)
    
    viz_samples_ds = ImageDataset(image_files=samples_of_interest_df[modality].values, labels=samples_of_interest_df[targets].values, transform=viz_samples_of_interest_transform, image_only=False) # type: ignore
    viz_samples_loader = DataLoader(viz_samples_ds, batch_size=1, shuffle=False, num_workers=1)

    print(f"Loading {bg_samples} background samples...")

    bg_batch = next(iter(bg_loader))
    bg_volumes, bg_labels, bg_metadata = bg_batch

    model.eval()

    e = shap.DeepExplainer(model, bg_volumes)

    print(f"E(x) = {e.expected_value[0]}")

    print(f"Making output directory {output_dir}...")
    
    os.makedirs(f"{output_dir}",exist_ok=True)

    print(f"Running DeepShap on {len(samples_ds)} samples...")
   
    for i, (inf_batch, viz_batch) in enumerate(zip(samples_loader, viz_samples_loader)):
        test_volume, test_label, test_metadata = inf_batch
        viz_volume, _, _ = viz_batch
        subjid = test_metadata["filename_or_obj"][0].split("/")[-1].split(".")[0]
        output_path = os.path.join(output_dir, subjid)
        print(f"{i+1}/{len(samples_ds)} {subjid}", flush=True)
        try:
            shap_values = e.shap_values(test_volume, check_additivity=False)
            delta = check_shap_additivity(e, model, test_volume, [shap_values.squeeze(-1)])
            save_shap_maps(output_path, viz_volume, shap_values, delta)
        except Exception as ex:
            with open(f"{output_path}.err", "w") as f:
                f.write(str(ex))
