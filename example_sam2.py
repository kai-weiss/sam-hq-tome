import os

import torch

from evaluate import EvaluateArgs2
from flops import get_flops_sam2

if __name__ == '__main__':

    # All testcases: [None, "tome_", "grad_tome_", "pitome_", "pitome_v1_", "pitome_v2_"]
    test_cases = [
        None, "tome_", "grad_tome_", "pitome_", "pitome_v1_", "pitome_v2_"
    ]

    for setting in test_cases:

        prefix = setting or ""
        cfg_path = f"configs/sam2.1/{prefix}sam2.1_hiera_l.yaml"
        # cfg_path = f"configs/sam2.1/{prefix}.yaml"

        evaluate_args = EvaluateArgs2(
            dataset="davis",
            output="",
            sam2_cfg=cfg_path,
            sam2_checkpoint="sam-hq2/checkpoints/sam2.1_hiera_large.pt",
            device="cuda",
            input_size=[1024, 1024],
            batch_size=1,
            multiple_masks=False,
            per_obj_png_file=False, # Set to True if you evaluating on the SA-V dataset
            apply_postprocessing=False,
            use_vos_optimized_video_predictor=False
        )

        flops_per_image = get_flops_sam2(args=evaluate_args)
        print(flops_per_image)
