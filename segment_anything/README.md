# Segment Anything in High Quality Installation Instructions

Note: this assumes that you followed the prerequisites [here](https://github.com/kai-weiss/sam-hq-tome/tree/main?tab=readme-ov-file#preparations).
Please look [here](https://github.com/SysCV/sam-hq/tree/main#standard-installation) for more information.



### Conda environment setup
```bash
conda activate sam-hq-tome
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install opencv-python pycocotools matplotlib onnxruntime onnx timm

# under your working directory
pip install -e .
export PYTHONPATH=$(pwd)
```

### **Model Checkpoints**

Create a folder, named `checkpoints`, under the root of this repository.
Init checkpoint can be downloaded from [hugging face link](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/pretrained_checkpoint)


### Expected checkpoint overview

```
pretrained_checkpoint
|____sam_vit_l_maskdecoder.pth
|____sam_vit_l_0b3195.pth
```
