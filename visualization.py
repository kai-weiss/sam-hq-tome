import os
import random
from dataclasses import dataclass
from typing import Optional, List

import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt, patches
from skimage import io, measure

from evaluate import EvaluateArgs
from maskdecoderhq import MaskDecoderHQExample
from segment_anything import SamAutomaticMaskGenerator
from tome_sam.build_tome_sam import tome_sam_model_registry
from tome_sam.utils import misc
from tome_sam.utils.tome_presets import SAMToMeSetting, ToMeConfig, PiToMe, ToMe, ToMeSD
import torch.nn.functional as F

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import warnings
warnings.filterwarnings("ignore")


@dataclass
class VisualizeArgs:
    input_image: str
    input_mask: str
    output: str
    model_type: str
    checkpoint: str
    seed: int
    input_size: List[int]
    tome_setting: Optional[SAMToMeSetting] = None

def plot_image_mask_bbox(image, pred_mask, gt_mask, bounding_box, save_path='output.png'):
    """
    Visualize an image with an overlayed mask and bounding box
    Args:
        image(torch.Tensor): (3, H, W)
        pred_mask(torch.Tensor): (1, H, W), with boolean values
        gt_mask(torch.Tensor): (1, H, W), with boolean values
        bounding_box(torch.Tensor): (1, 4), [x_min, y_min, x_max, y_max]
        save_path: path to save the output image
    """
    image = image.permute(1, 2, 0).numpy().astype(np.uint8)
    pred_mask = pred_mask.squeeze(0).numpy()
    gt_mask = gt_mask.squeeze(0).numpy()
    bbox = bounding_box.squeeze(0).numpy()

    fig, ax = plt.subplots(figsize=(8, 8))

    # overlay masks
    ax.imshow(image)
    ax.imshow(pred_mask, cmap='Reds', alpha=0.2)
    ax.imshow(gt_mask, cmap='Greens', alpha=0.2)

    pred_contours = measure.find_contours(pred_mask, 0.5)
    gt_contours = measure.find_contours(gt_mask, 0.5)

    # draw contours
    for contour in pred_contours:
        ax.plot(contour[:, 1], contour[:, 0], 'red', linewidth=1, label='pred_mask')

    for contour in gt_contours:
        ax.plot(contour[:, 1], contour[:, 0], 'darkgreen', linewidth=1, label='gt_mask')

    # bounding boxes
    x_min, y_min, x_max, y_max = bbox
    bbox_rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=2, edgecolor='black', facecolor='none'
    )
    ax.add_patch(bbox_rect)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Segmentation output image saved to {save_path}')


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def visualize_automatic_mask_generator(args: VisualizeArgs, original_resolution=False):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tome_sam = tome_sam_model_registry[args.model_type](
        checkpoint=args.checkpoint,
        tome_setting=args.tome_setting,
    )

    mask_generator = SamAutomaticMaskGenerator(model=tome_sam,
                                               points_per_side=32,
                                                pred_iou_thresh=0.86,
                                                stability_score_thresh=0.92,
                                                crop_n_layers=1,
                                                crop_n_points_downscale_factor=2,
                                                min_mask_region_area=100)
    image = cv2.imread(args.input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(args.output, bbox_inches='tight', dpi=300)
    plt.close(fig)




def visualize_output_mask(args: VisualizeArgs, original_resolution=False):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tome_sam = tome_sam_model_registry[args.model_type](
        checkpoint=args.checkpoint,
        tome_setting=args.tome_setting,
    )

    tome_sam.eval()

    im = io.imread(args.input_image)
    gt = io.imread(args.input_mask)
    if len(gt.shape) > 2:
        gt = gt[:, :, 0]
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    im = torch.tensor(im.copy(), dtype=torch.float32)
    im = torch.transpose(torch.transpose(im, 1, 2), 0, 1) # (3, H, W)
    _, original_H, original_W = im.shape
    gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0) # (1, H, W)

    # Resize
    resized_im = torch.squeeze(F.interpolate(torch.unsqueeze(im, 0), args.input_size, mode='bilinear'), dim=0)
    resized_gt= torch.squeeze(F.interpolate(torch.unsqueeze(gt, 0), args.input_size, mode='bilinear'), dim=0) # (1, H, W)

    resized_bounding_box = misc.masks_to_boxes(resized_gt[0].unsqueeze(0)) # (1, 4)

    dict_input = dict()
    dict_input['image'] = resized_im.to(torch.uint8)
    dict_input['boxes'] = resized_bounding_box
    dict_input['original_size'] = resized_im.shape[1:]

    with torch.no_grad():
        resized_mask = tome_sam([dict_input], multimask_output=False)[0][0]['masks'][0] # (1, H, W)

    # evaluation on original resolution
    mask = torch.squeeze(F.interpolate(torch.unsqueeze(resized_mask.float(), 0), [original_H, original_W], mode='bilinear'), dim=0)
    bounding_box = misc.masks_to_boxes(gt[0].unsqueeze(0))
    m_iou = misc.mask_iou(mask, gt)
    b_iou = misc.boundary_iou(gt, mask)
    print(f'Mask IoU: {m_iou}, Boundary IoU: {b_iou}')
    if original_resolution:
        plot_image_mask_bbox(im, mask, gt, bounding_box, save_path=args.output)
    else:
        plot_image_mask_bbox(resized_im, resized_mask, resized_gt, resized_bounding_box, save_path=args.output)

def visualize_output_mask_sam2(args: VisualizeArgs, cfg, ckpt, original_resolution=False):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    sam2_core = build_sam2(cfg, ckpt).eval().cuda()
    predictor_sam2 = SAM2ImagePredictor(sam2_core)

    im = io.imread(args.input_image)
    gt = io.imread(args.input_mask)
    if len(gt.shape) > 2:
        gt = gt[:, :, 0]
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    im = torch.tensor(im.copy(), dtype=torch.float32)
    im = torch.transpose(torch.transpose(im, 1, 2), 0, 1) # (3, H, W)
    _, original_H, original_W = im.shape
    gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0) # (1, H, W)

    # Resize
    resized_im = torch.squeeze(F.interpolate(torch.unsqueeze(im, 0), args.input_size, mode='bilinear'), dim=0)
    resized_gt= torch.squeeze(F.interpolate(torch.unsqueeze(gt, 0), args.input_size, mode='bilinear'), dim=0) # (1, H, W)

    resized_bounding_box = misc.masks_to_boxes(resized_gt[0].unsqueeze(0)) # (1, 4)

    # ----- SAM 2 replacement ----------------
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        predictor_sam2.set_image(resized_im.permute(1, 2, 0).cpu().numpy())
        # choose one prompt type you like
        masks, _, _ = predictor_sam2.predict(
            multimask_output=False
        )
        resized_mask = torch.from_numpy(masks[0]).unsqueeze(0)  # (1, H, W)

    # evaluation on original resolution
    mask = torch.squeeze(F.interpolate(torch.unsqueeze(resized_mask.float(), 0), [original_H, original_W], mode='bilinear'), dim=0)
    bounding_box = misc.masks_to_boxes(gt[0].unsqueeze(0))
    m_iou = misc.mask_iou(mask, gt)
    b_iou = misc.boundary_iou(gt, mask)
    print(f'Mask IoU: {m_iou}, Boundary IoU: {b_iou}')
    if original_resolution:
        plot_image_mask_bbox(im, mask, gt, bounding_box, save_path=args.output)
    else:
        plot_image_mask_bbox(resized_im, resized_mask, resized_gt, resized_bounding_box, save_path=args.output)

if __name__ == '__main__':

    # Parameters for SAM2
    ckpt = "sam-hq2/checkpoints/sam2.1_hiera_large.pt"
    cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Activate SAM-HQ
    samHQ = True

    # Define which transformer layers to apply ToMe on
    # common_layers = [0, 4, 5, 10, 16, 17, 19, 20, 22, 23]
    common_layers = [0, 4, 5, 10, 16, 17, 19, 20, 22, 23]

    # Shared parameters
    r = 0.5
    margin = 0.5
    alpha = 1.0

    # Settings for different ToMe variants, all using the same layers
    tome_setting: SAMToMeSetting = {
        l: ToMeConfig(mode='tome', params=ToMe(r=r))
        for l in common_layers
    }

    grad_tome_setting: SAMToMeSetting = {
        l: ToMeConfig(mode='grad_tome', params=ToMe(r=r))
        for l in common_layers
    }

    pitome_setting: SAMToMeSetting = {
        l: ToMeConfig(mode='pitome', params=PiToMe(r=r, margin=margin, alpha=alpha))
        for l in common_layers
    }

    pitome_setting_v1: SAMToMeSetting = {
        l: ToMeConfig(mode='pitome_v1', params=PiToMe(r=r, margin=margin, alpha=alpha))
        for l in common_layers
    }

    pitome_setting_v2: SAMToMeSetting = {
        l: ToMeConfig(mode='pitome_v2', params=PiToMe(r=r, margin=margin, alpha=alpha))
        for l in common_layers
    }

    init = False
    args = VisualizeArgs(
        input_image='data/DIS5K/DIS-VD/im/7#Electrical#1#Cable#3142447262_1f6832e91c_o.jpg',
        input_mask='data/DIS5K/DIS-VD/gt/7#Electrical#1#Cable#3142447262_1f6832e91c_o.png',
        # input_image='data/DAVIS/JPEGImages/Full-Resolution/camel/00024.jpg',
        # input_mask='data/DAVIS/Annotations/Full-Resolution/camel/00024.png',
        output='sam2_none.png',
        model_type="vit_l",
        checkpoint="checkpoints/sam_vit_l_0b3195.pth",
        seed=42,
        input_size=[1024, 1024],
        tome_setting=None,
    )

    evaluate_args = EvaluateArgs(
        dataset="dis",  # choose between dis, thin, hrsod, coift
        output="",
        model_type="vit_l",
        checkpoint="checkpoints/sam_vit_l_0b3195.pth",
        device="cuda",
        seed=42,
        input_size=[1024, 1024],
        batch_size=1,
        world_size=1,
        dist_url='env://',
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        rank=0,
        multiple_masks=False,
        restore_model="work_dirs/hq_sam_l/epoch_11.pth",
        tome_setting=None,
    )

    if samHQ:
        print("--- init Maskloader HQ ---")
        gpu = evaluate_args.local_rank

        if not init:
            misc.init_distributed_mode(evaluate_args)
            init = True

        net = MaskDecoderHQExample(evaluate_args.model_type)
        if torch.cuda.is_available():
            net.cuda()
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu],
                                                        find_unused_parameters=False)
        net_without_ddp = net.module

        if torch.cuda.is_available():
            net_without_ddp.load_state_dict(torch.load(evaluate_args.restore_model))
        else:
            net_without_ddp.load_state_dict(torch.load(evaluate_args.restore_model, map_location="cpu"))

    # visualize_output_mask_sam2(args=args, cfg=cfg, ckpt=ckpt)
    visualize_output_mask(args)
    # visualize_automatic_mask_generator(args)