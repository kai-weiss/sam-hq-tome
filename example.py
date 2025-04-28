import os

import torch

from evaluate import EvaluateArgs, evaluate, evaluate_hq
from maskdecoderhq import MaskDecoderHQExample
from tome_sam.utils import misc
from tome_sam.utils.tome_presets import SAMToMeSetting, ToMeSD, ToMeConfig, PiToMe, ToMe
from flops import get_flops, get_flops_hq

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

if __name__ == '__main__':
    # All testcases: [None, tome_setting, grad_tome_setting, pitome_setting, pitome_setting_v1, pitome_setting_v2]
    test_cases = [None, tome_setting, grad_tome_setting, pitome_setting, pitome_setting_v1, pitome_setting_v2]

    init = False
    for setting in test_cases:
        evaluate_args = EvaluateArgs(
            dataset="coift",  # choose between dis, thin, hrsod, coift
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
            tome_setting=setting,
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

        if samHQ:
            eval_results = evaluate_hq(args=evaluate_args, net=net)
        else:
            eval_results = evaluate(args=evaluate_args)
        print(eval_results)

        # if samHQ:
        #    flops_per_image = get_flops_hq(args=evaluate_args, net=net)
        # else:
        #    flops_per_image = get_flops(args=evaluate_args)
        # print(flops_per_image)
