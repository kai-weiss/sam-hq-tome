import json
import os
import random

import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm

from evaluate import parse_and_convert_args, EvaluateArgs, EvaluateArgs2, dataset_name_mapping
from tome_sam.build_tome_sam import tome_sam_model_registry
from tome_sam.utils import misc
from tome_sam.utils.dataloader import get_im_gt_name_dict, create_dataloaders, Resize, gather_davis_paths
from fvcore.nn import FlopCountAnalysis
import torch.nn.functional as F

from tome_sam.utils.json_serialization import convert_to_serializable_dict

import warnings

warnings.filterwarnings("ignore")


def get_flops(args: EvaluateArgs) -> dict:
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # gflops_sam = []
    gflops_image_encoder = []

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    ### Create eval dataloader ###
    print(f"--- Create valid dataloader with dataset {args.dataset} ---")
    dataset_info = dataset_name_mapping[args.dataset]
    valid_im_gt_path = get_im_gt_name_dict(dataset_info, flag='valid')
    valid_dataloader, valid_dataset = create_dataloaders(valid_im_gt_path,
                                                         my_transforms=[
                                                             Resize(args.input_size),
                                                         ],
                                                         batch_size=args.batch_size,
                                                         training=False)
    print(f"--- Valid dataloader with dataset {args.dataset} created ---")

    ### Create model with specified arguments ###
    print(f"--- Create SAM {args.model_type} with token merging in layers {args.tome_setting} ---")

    tome_sam = tome_sam_model_registry[args.model_type](
        checkpoint=args.checkpoint,
        tome_setting=args.tome_setting,
    )
    tome_sam.to(device)
    tome_sam.eval()

    ### Start Flop count analysis ###
    print(f"--- Start flop count analysis ---")
    for data_val in tqdm(valid_dataloader, position=0, leave=False):
        imidx, inputs, labels, shapes, labels_ori = data_val["imidx"], data_val["image"], data_val["label"], data_val[
            "shape"], data_val["ori_label"]

        inputs = inputs.to(device)
        labels = labels.to(device)

        # (B, C, H, W) -> (B, H, W, C)
        imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()

        labels_box = misc.masks_to_boxes(labels[:, 0, :, :])
        batched_input = []

        for b_i in range(len(imgs)):
            dict_input = dict()
            input_image = torch.as_tensor(imgs[b_i].astype(np.uint8), device=device).permute(2, 0,
                                                                                             1).contiguous()  # (C, H, W)
            dict_input['image'] = input_image
            dict_input['boxes'] = labels_box[b_i: b_i + 1]
            dict_input['original_size'] = imgs[b_i].shape[:2]
            batched_input.extend([dict_input])

        """
        # flops evaluation on whole sam
        with torch.no_grad():
            # because batched_input is a list of dictionary, not a normally expected tensor input, which is required for flops count
            flops = FlopCountAnalysis(tome_sam, (batched_input, args.multiple_masks))
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            gflops_sam.append((flops.total()/1e9)/args.batch_size)
        """

        tome_sam.to(device)
        image_encoder = tome_sam.image_encoder
        input_images = torch.stack([tome_sam.preprocess(x['image']) for x in batched_input], dim=0).to(device)
        # flops evaluation only on image encoder
        with torch.no_grad():
            flops = FlopCountAnalysis(image_encoder, input_images)
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            gflops_image_encoder.append((flops.total() / 1e9) / args.batch_size)

    # sam_flops_per_image = np.mean(gflops_sam)
    image_encoder_flops_per_image = np.mean(gflops_image_encoder)

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        filename = os.path.join(args.output, 'flops.json')
        with open(filename, 'w') as f:
            json.dump({
                # 'flops/img(sam)': str(sam_flops_per_image),
                'flops/img(image_encoder)': str(image_encoder_flops_per_image),
                'evaluate_args': convert_to_serializable_dict(args),
            }, f, indent=4, default=str)

    return {
        # 'flops/img(sam)': sam_flops_per_image,
        'flops/img(image_encoder)': image_encoder_flops_per_image
    }


def get_flops_hq(net, args: EvaluateArgs) -> dict:
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # gflops_sam = []
    gflops_image_encoder = []

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    ### Create eval dataloader ###
    print(f"--- Create valid dataloader with dataset {args.dataset} ---")
    dataset_info = dataset_name_mapping[args.dataset]
    valid_im_gt_path = get_im_gt_name_dict(dataset_info, flag='valid')
    valid_dataloader, valid_dataset = create_dataloaders(valid_im_gt_path,
                                                         my_transforms=[
                                                             Resize(args.input_size),
                                                         ],
                                                         batch_size=args.batch_size,
                                                         training=False)
    print(f"--- Valid dataloader with dataset {args.dataset} created ---")

    ### Create model with specified arguments ###
    print(f"--- Create SAM {args.model_type} with token merging in layers {args.tome_setting} ---")

    tome_sam = tome_sam_model_registry[args.model_type](
        checkpoint=args.checkpoint,
        tome_setting=args.tome_setting,
    )
    tome_sam.to(device)
    tome_sam.eval()

    net.eval()

    ### Start Flop count analysis ###
    print(f"--- Start flop count analysis ---")
    for data_val in tqdm(valid_dataloader, position=0, leave=False):
        imidx, inputs, labels, shapes, labels_ori = data_val["imidx"], data_val["image"], data_val["label"], data_val[
            "shape"], data_val["ori_label"]

        inputs = inputs.to(device)
        labels = labels.to(device)

        # (B, C, H, W) -> (B, H, W, C)
        imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()

        labels_box = misc.masks_to_boxes(labels[:, 0, :, :])
        batched_input = []

        for b_i in range(len(imgs)):
            dict_input = dict()
            input_image = torch.as_tensor(imgs[b_i].astype(np.uint8), device=device).permute(2, 0,
                                                                                             1).contiguous()  # (C, H, W)
            dict_input['image'] = input_image
            dict_input['boxes'] = labels_box[b_i: b_i + 1]
            dict_input['original_size'] = imgs[b_i].shape[:2]
            batched_input.extend([dict_input])

        """
        # flops evaluation on whole sam
        with torch.no_grad():
            # because batched_input is a list of dictionary, not a normally expected tensor input, which is required for flops count
            flops = FlopCountAnalysis(tome_sam, (batched_input, args.multiple_masks))
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            gflops_sam.append((flops.total()/1e9)/args.batch_size)
        """

        tome_sam.to(device)
        image_encoder = tome_sam.image_encoder
        input_images = torch.stack([tome_sam.preprocess(x['image']) for x in batched_input], dim=0).to(device)
        # flops evaluation only on image encoder
        with torch.no_grad():
            flops = FlopCountAnalysis(image_encoder, input_images)
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            gflops_image_encoder.append((flops.total() / 1e9) / args.batch_size)

    # sam_flops_per_image = np.mean(gflops_sam)
    image_encoder_flops_per_image = np.mean(gflops_image_encoder)

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        filename = os.path.join(args.output, 'flops.json')
        with open(filename, 'w') as f:
            json.dump({
                # 'flops/img(sam)': str(sam_flops_per_image),
                'flops/img(image_encoder)': str(image_encoder_flops_per_image),
                'evaluate_args': convert_to_serializable_dict(args),
            }, f, indent=4, default=str)

    return {
        # 'flops/img(sam)': sam_flops_per_image,
        'flops/img(image_encoder)': image_encoder_flops_per_image
    }



def get_flops_sam2(args: EvaluateArgs2) -> dict:

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Prepare one small dataloader to get a single batch
    dataset_info = dataset_name_mapping[args.dataset]
    valid_im_gt_path = gather_davis_paths(dataset_info, flag='valid')
    valid_dataloader, _ = create_dataloaders(
        valid_im_gt_path,
        my_transforms=[Resize(args.input_size)],
        batch_size=args.batch_size,
        training=False
    )

    # grab exactly one batch (no per-frame loop)
    batch = next(iter(valid_dataloader))
    imgs = batch["image"].to(device)  # shape: [B, 3, H, W]

    # Build SAM2 predictor and extract its image encoder
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=[f"++model.non_overlap_masks={'false' if args.per_obj_png_file else 'true'}"],
        vos_optimized=args.use_vos_optimized_video_predictor,
    )
    predictor.to(device).eval()

    gflops_enc = []
    gflops_mem = []
    gflops_mask = []

    # Do one FLOPs trace and compute GFLOPs per image
    with torch.no_grad():
        # --- image encoder FLOPs ---
        enc = predictor.image_encoder
        fa_enc = FlopCountAnalysis(enc, imgs).unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        gflops_enc.append((fa_enc.total() / 1e9) / imgs.size(0))

        # --- memory attention FLOPs ---
        backbone_out = predictor.forward_image(imgs)
        _, vision_feats, vision_pos, feat_sizes = predictor._prepare_backbone_features(backbone_out)
        curr = [vision_feats[-1]]
        curr_pos = [vision_pos[-1]]

        H, W = feat_sizes[-1]
        mem_tokens = max(1, H * W * predictor.num_maskmem)
        mem_dim = predictor.mem_dim

        # Use CPU to avoid potential GPU kernel issues during the FLOP trace
        mem_device = torch.device("cpu")
        predictor.memory_attention.to(mem_device)
        memory = torch.randn(mem_tokens, imgs.size(0), mem_dim, device=mem_device, dtype=curr[0].dtype)
        memory_pos = torch.randn_like(memory)
        curr_cpu = [c.to(mem_device) for c in curr]
        curr_pos_cpu = [c.to(mem_device) for c in curr_pos]

        fa_mem = FlopCountAnalysis(
            predictor.memory_attention,
            (curr_cpu, memory, curr_pos_cpu, memory_pos, 0),
        ).unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        gflops_mem.append((fa_mem.total() / 1e9) / imgs.size(0))

        # --- mask decoder FLOPs ---
        mask_device = torch.device("cpu")
        predictor.sam_mask_decoder.to(mask_device)
        predictor.sam_prompt_encoder.to(mask_device)

        # create random fused feature with correct shape
        pix_feat = torch.randn(
            imgs.size(0), predictor.hidden_dim, H, W, device=mask_device
        )

        # high-res features if available
        if len(feat_sizes) > 1:
            high_res_features = [
                torch.randn(
                    imgs.size(0),
                    predictor.hidden_dim,
                    s[0],
                    s[1],
                    device=mask_device,
                )
                for s in feat_sizes[:-1]
            ]
        else:
            high_res_features = None

        sam_point_coords = torch.zeros(
            imgs.size(0), 1, 2, device=mask_device
        )
        sam_point_labels = -torch.ones(
            imgs.size(0), 1, dtype=torch.int32, device=mask_device
        )
        sparse_emb, dense_emb = predictor.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=None,
        )
        dense_pe = predictor.sam_prompt_encoder.get_dense_pe().to(mask_device)

        fa_mask = (
            FlopCountAnalysis(
                predictor.sam_mask_decoder,
                (
                    pix_feat,
                    dense_pe,
                    sparse_emb,
                    dense_emb,
                    args.multiple_masks,
                    False,
                    high_res_features,
                ),
            )
            .unsupported_ops_warnings(False)
            .uncalled_modules_warnings(False)
        )
        gflops_mask.append((fa_mask.total() / 1e9) / imgs.size(0))

    out = {
        "flops/img(image_encoder)": float(sum(gflops_enc) / len(gflops_enc)),
        "flops/img(memory_attention)": float(sum(gflops_mem) / len(gflops_mem)),
        "flops/img(mask_decoder)": float(sum(gflops_mask) / len(gflops_mask)),
    }

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "flops.json"), "w") as f:
            json.dump({**out, "evaluate_args": convert_to_serializable_dict(args)}, f, indent=4, default=str)

    return out


if __name__ == "__main__":
    args = parse_and_convert_args()
    avg_flops = get_flops(args)
    print(avg_flops)
