## Optimizing SAM and SAM-2 Foundation Models: A Token Merging Approach

### Installation
Refer to installation steps of the [Segment Anything repository](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#installation).

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions here to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install `tome_sam` repository and install with:
```
git clone https://github.com/xxjsw/tome_sam.git
cd tome_sam
pip install -e.
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. jupyter is also required to run the example notebooks.
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

### Download Checkpoints
Create a directory `checkpoints` directly inside the root directory and download the pre-trained SAM checkpoints into it:
```
# ViT-B 
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# ViT-L 
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# ViT-H 
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Data Preparation

### Datasets for SAM/HQ-SAM

Refer to dataset preparation from [Segment Anything in high Quality](https://github.com/SysCV/sam-hq/blob/main/train/README.md#1-data-preparation).

Download the datasets DIS5k.zip and thin_object_detection.zip from this [Hugging Face link](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/data).

### Datasets for SAM 2

Refer to dataset preparation from [SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2/tree/main/sav_dataset).

Download the following three datasets: [DAVIS](https://davischallenge.org/index.html), [MOSE](https://henghuiding.github.io/MOSE/), [SA-V](https://ai.meta.com/datasets/segment-anything-video-downloads/) 

###  Overview
At the end, the data folder should have the following structure:
```
sam-hq-tome
|____data
   |____DAVIS
   |____DIS5K
   |____MOSE
   |____SA-V
   |____thin_object_detection
   | |____COIFT
   | |____HRSOD
   | |____ThinObject5K
```
# Evaluation

## Evaluation on SAM/HQ-SAM

The script `example.py` evaluates SAM  with various ToMe (Token Merging) variants, optionally using the HQ Mask Decoder for HQ-SAM. 
It runs multiple test cases (different ToMe configurations iterating in a loop) and reports evaluation metrics and FLOPs.
The default settings are the same as chosen in the paper:
```
# Activate HQ mask decoder (SAM-HQ)
samHQ = True

# Transformer layers to apply ToMe on
common_layers = [0, 4, 5, 10, 16, 17, 19, 20, 22, 23]

# Shared parameters
r = 0.5

# PiToMe parameters
margin = 0.5
alpha = 1.0

# Define ToMe settings for each variant
tome_setting: SAMToMeSetting = { ... }
grad_tome_setting: SAMToMeSetting = { ... }
pitome_setting: SAMToMeSetting = { ... }
pitome_setting_v1: SAMToMeSetting = { ... }
pitome_setting_v2: SAMToMeSetting = { ... }
```

Optionally it is also possible to define your own setting and "mix-and-match" different token merging strategies, for example:
```
new_setting: SAMToMeSetting = {
    2: ToMeConfig(
        mode='pitome',
        params=ToMe(r=0.5)
    ),
    5: ToMeConfig(
        mode='pitome',
        params=PiToMe(r=0.5, margin=0.0, alpha=1.0)
    ),
}
```
Make sure to include your setting in the variable `test_cases`. 



#### Configurable Parameters Overview for `example.py`:
| Parameter           | Description                                                          | Default                                                                           |
|---------------------|----------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `samHQ`             | Whether to use the HQ Mask Decoder (`evaluate_hq` path)             | `True`                                                                            |
| `common_layers`     | List of transformer layer indices to apply ToMe on                   | `[0, 4, 5, 10, 16, 17, 19, 20, 22, 23]`                                            |
| `r`                 | Reduction ratio for ToMe token merging                               | `0.5`                                                                             |
| `margin`            | Margin parameter for PiToMe                                          | `0.5`                                                                             |
| `alpha`             | Alpha parameter for PiToMe                                           | `1.0`                                                                             |
| `dataset`           | Dataset to evaluate (choices: `dis`, `thin`, `hrsod`, `coift`)        | Set in `EvaluateArgs(dataset=...)`, default shown in code is `"coift"`            |
| `output`            | Output directory for results                                         | `"./outputs/"`                                                                  |
| `model_type`        | SAM backbone variant (e.g., `vit_l`, `vit_b`)                        | `"vit_l"`                                                                       |
| `checkpoint`        | Path to SAM checkpoint file                                          | `"checkpoints/sam_vit_l_0b3195.pth"`                                            |
| `device`            | Compute device (e.g., `cuda`, `cpu`)                                 | `"cuda"`                                                                        |
| `seed`              | Random seed                                                          | `42`                                                                              |
| `input_size`        | Input image size (`[height, width]`)                                 | `[1024, 1024]`                                                                    |
| `batch_size`        | Batch size per GPU                                                   | `1`                                                                               |
| `world_size`        | Number of processes for distributed evaluation                       | `1`                                                                               |
| `dist_url`          | URL for initializing distributed mode                                | `'env://'`                                                                        |
| `local_rank`        | Local GPU rank (from `LOCAL_RANK` env var)                           | `int(os.environ.get("LOCAL_RANK", 0))`                                          |
| `rank`              | Global process rank                                                  | `0`                                                                               |
| `multiple_masks`    | Whether to predict multiple masks per prompt                         | `False`                                                                           |
| `restore_model`     | Path to the trained Mask Decoder HQ model checkpoint                 | `"work_dirs/hq_sam_l/epoch_11.pth"`                                              |
| `tome_setting`      | ToMe configuration dictionary for the current test case              | One of the settings defined at the top                                           |



## Evaluation on SAM 2

### Accuracy Evaluation (J&F)
To run the accuracy evaluation, navigate to:
```
cd sam-hq-tome/sam-hq2
```

Please also refer to [SAM 2 toolkits](https://github.com/facebookresearch/sam2/tree/main/tools) for more information.


### FLOPS Evaluation
To run the FLOPS evaluation on every token merging variant, execute `example_sam2.py`.
The default settings are the same as chosen in the paper, so nothing has to be changed.

#### Configurable Parameters Overview for `example_sam2.py`:
| Parameter               | Description                                                                                      | Default                                                        |
|-------------------------|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| `test_cases`            | List of model variants prefixes to test (None, "tome_", "grad_tome_", etc.)                   | `[None, "tome_", "grad_tome_", "pitome_", "pitome_v1_", "pitome_v2_"]` |
| `cfg_path`              | Path to the SAM2.1 configuration YAML file, built as `configs/sam2.1/{prefix}sam2.1_hiera_l.yaml` | `"configs/sam2.1/sam2.1_hiera_l.yaml"`                        |
| `dataset`               | Dataset for evaluation (via `EvaluateArgs2`)                                                     | `"davis"`                                                     |
| `output`                | Directory or prefix for output results                                                           | `""` (empty string)                                           |
| `sam2_cfg`              | Full path to the SAM2.1 config used by `EvaluateArgs2`                                            | `cfg_path` variable                                            |
| `sam2_checkpoint`       | Path to the SAM2.1 model checkpoint file                                                         | `"sam-hq2/checkpoints/sam2.1_hiera_large.pt"`                |
| `device`                | Compute device (e.g., `cuda`, `cpu`)                                                             | `"cuda"`                                                     |
| `input_size`            | Input resolution for FLOPs computation (`[height, width]`)                                       | `[1024, 1024]`                                                |
| `batch_size`            | Batch size per GPU process                                                                       | `1`                                                            |
| `multiple_masks`        | Whether to predict multiple masks per prompt                                                     | `False`                                                        |
| `per_obj_png_file`      | Whether to save per-object PNG files for SA-V evaluation                                         | `False`                                                        |
| `apply_postprocessing`  | Whether to run postprocessing on predictions                                                     | `False`                                                        |
| `use_vos_optimized_video_predictor` | Whether to use a VOS-optimized video predictor                                              | `False`                                                        |

   
