import os

import torch

from evaluate import EvaluateArgs, evaluate, evaluate_hq
from maskdecoderhq import MaskDecoderHQExample
from tome_sam.utils import misc
from tome_sam.utils.tome_presets import SAMToMeSetting, ToMeSD, ToMeConfig, PiToMe, ToMe
from flops import get_flops

tome_setting: SAMToMeSetting = {
    7: ToMeConfig(
        mode='tome',
        params=ToMe(r=0.5)
    ),
    8: ToMeConfig(
        mode='tome',
        params=ToMe(r=0.5)
    ),
    9: ToMeConfig(
        mode='tome',
        params=ToMe(r=0.5)
    ),
    10: ToMeConfig(
        mode='tome',
        params=ToMe(r=0.5)
    ),
    11: ToMeConfig(
        mode='tome',
        params=ToMe(r=0.5)
    ),
}

grad_tome_setting: SAMToMeSetting = {
    7: ToMeConfig(
        mode='grad_tome',
        params=ToMe(r=0.5)
    ),
    8: ToMeConfig(
        mode='grad_tome',
        params=ToMe(r=0.5)
    ),
    9: ToMeConfig(
        mode='grad_tome',
        params=ToMe(r=0.5)
    ),
    10: ToMeConfig(
        mode='grad_tome',
        params=ToMe(r=0.5)
    ),
    11: ToMeConfig(
        mode='grad_tome',
        params=ToMe(r=0.5)
    ),
}

pitome_setting: SAMToMeSetting = {
    7: ToMeConfig(
        mode='pitome',
        params=PiToMe(r=0.5, margin=0.5, alpha=1.0)
    ),
    8: ToMeConfig(
        mode='pitome',
        params=PiToMe(r=0.5, margin=0.5, alpha=1.0)
    ),
    9: ToMeConfig(
        mode='pitome',
        params=PiToMe(r=0.5, margin=0.5, alpha=1.0)
    ),
    10: ToMeConfig(
        mode='pitome',
        params=PiToMe(r=0.5, margin=0.5, alpha=1.0)
    ),
    11: ToMeConfig(
        mode='pitome',
        params=PiToMe(r=0.5, margin=0.5, alpha=1.0)
    ),
}

pitome_setting_v1: SAMToMeSetting = {
    7: ToMeConfig(
        mode='pitome_v1',
        params=PiToMe(r=0.5, margin=0.5, alpha=1.0)
    ),
    8: ToMeConfig(
        mode='pitome_v1',
        params=PiToMe(r=0.5, margin=0.5, alpha=1.0)
    ),
    9: ToMeConfig(
        mode='pitome_v1',
        params=PiToMe(r=0.5, margin=0.5, alpha=1.0)
    ),
    10: ToMeConfig(
        mode='pitome_v1',
        params=PiToMe(r=0.5, margin=0.5, alpha=1.0)
    ),
    11: ToMeConfig(
        mode='pitome_v1',
        params=PiToMe(r=0.5, margin=0.5, alpha=1.0)
    ),
}

pitome_setting_v2: SAMToMeSetting = {
    7: ToMeConfig(
        mode='pitome_v2',
        params=PiToMe(r=0.5, margin=0.5, alpha=1.0)
    ),
    8: ToMeConfig(
        mode='pitome_v2',
        params=PiToMe(r=0.5, margin=0.5, alpha=1.0)
    ),
    9: ToMeConfig(
        mode='pitome_v2',
        params=PiToMe(r=0.5, margin=0.5, alpha=1.0)
    ),
    10: ToMeConfig(
        mode='pitome_v2',
        params=PiToMe(r=0.5, margin=0.5, alpha=1.0)
    ),
    11: ToMeConfig(
        mode='pitome_v2',
        params=PiToMe(r=0.5, margin=0.5, alpha=1.0)
    ),
}


if __name__ == '__main__':
    # All testcases: [None, tome_setting, grad_tome_setting, pitome_setting, pitome_setting_v1, pitome_setting_v2]
    test_cases = [grad_tome_setting, pitome_setting, pitome_setting_v1, pitome_setting_v2]

    init = False
    for setting in test_cases:
        evaluate_args = EvaluateArgs(
            dataset="dis",
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

        # print("--- create valid dataloader ---")
        # valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
        # valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
        #                                                        my_transforms=[
        #                                                            Resize(EvaluateArgs.input_size)
        #                                                        ],
        #                                                        batch_size=EvaluateArgs.batch_size,
        #                                                        training=False)

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

        # eval_results = evaluate(args=evaluate_args)
        print(f"--- start Eval HQ with GPU {gpu} ---")
        eval_results = evaluate_hq(args=evaluate_args, net=net)
        print(eval_results)

        flops_per_image = get_flops(evaluate_args)
        print(flops_per_image)
