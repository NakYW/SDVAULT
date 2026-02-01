import argparse
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import copy
import pickle
import time
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import torchvision.transforms as transforms

feat_maps = []


def load_img(path, target_h: int, target_w: int):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")

    # Center-crop to target aspect ratio, then resize to (target_w, target_h)
    target_ratio = float(target_w) / float(target_h)
    input_ratio = float(x) / float(y)

    if input_ratio > target_ratio:
        # input is wider; crop width
        new_w = int(round(y * target_ratio))
        new_h = y
        left = max(0, (x - new_w) // 2)
        top = 0
    else:
        # input is taller; crop height
        new_w = x
        new_h = int(round(x / target_ratio))
        left = 0
        top = max(0, (y - new_h) // 2)

    right = min(x, left + new_w)
    bottom = min(y, top + new_h)
    image = image.crop((left, top, right, bottom))
    image = image.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sty", default="./data/sty", help="Path to style images directory")
    parser.add_argument("--ddim_inv_steps", type=int, default=50, help="DDIM inversion steps")
    parser.add_argument("--save_feat_steps", type=int, default=50, help="Save feature steps")
    parser.add_argument("--start_step", type=int, default=49, help="Start step")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta")
    parser.add_argument("--C", type=int, default=4, help="Latent channels")
    parser.add_argument("--f", type=int, default=8, help="Downsampling factor")
    parser.add_argument(
        "--H", type=int, 
        default=512, 
        help="Image height"
    )
    parser.add_argument(
        "--W", type=int, 
        default=512, 
        help="Image width"
    )
    parser.add_argument(
        "--T",
        type=float,
        default=1.1,
        help="attention temperature scaling hyperparameter",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.80, help="query preservation hyperparameter"
    )
    parser.add_argument(
        "--attn_layer",
        type=str,
        default="6,7,8,9,10,11",
        help="Injection attention feature layers",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="models/ldm/v1-inference.yaml",
        help="Model config",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-1.5/model.ckpt",
        help="Model checkpoint",
    )
    parser.add_argument(
        "--precomputed",
        type=str,
        default="./precomputed_feats",
        help="Save path for precomputed features",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="autocast",
        help='choices: ["full", "autocast"]',
    )

    opt = parser.parse_args()

    if opt.H % 32 != 0 or opt.W % 32 != 0:
        raise ValueError(
            f"H/W must be multiples of 32 to avoid UNet skip-size mismatches. "
            f"Got H={opt.H}, W={opt.W}."
        )

    feat_path_root = opt.precomputed

    seed_everything(22)
    os.makedirs(feat_path_root, exist_ok=True)

    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(",")))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    sampler.make_schedule(
        ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False
    )
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {}
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    global feat_maps
    feat_maps = [
        {
            "config": {
                "gamma": opt.gamma,
                "T": opt.T,
            }
        }
        for _ in range(50)
    ]

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map(xt, "z_enc", i)

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    # self-attn
                    attn = block[1].transformer_blocks[0].attn1
                    k_s = attn.K_s
                    v_s = attn.V_s
                    save_feature_map(k_s, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v_s, f"{feature_type}_{block_idx}_self_attn_v", i)
            block_idx += 1

    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks, i, "output_block")

    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]

        if isinstance(feature_map, torch.Tensor):
            tensor = feature_map.detach().to("cpu")
            tensor = tensor.to(torch.float16)
            feat_maps[cur_idx][filename] = tensor
        else:
            feat_maps[cur_idx][filename] = feature_map

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    sty_img_list = sorted(os.listdir(opt.sty))

    begin = time.time()
    print("Starting style feature extraction...")

    for sty_name in sty_img_list:
        sty_name_ = os.path.join(opt.sty, sty_name)
        init_sty = load_img(sty_name_, opt.H, opt.W).to(device)
        sty_feat_name = os.path.join(
            feat_path_root, os.path.basename(sty_name).split(".")[0] + "_sty.pkl"
        )

        if os.path.isfile(sty_feat_name):
            print(f"Feature file exists, skipping: {sty_feat_name}")
            continue

        print(f"Processing style image: {sty_name}")
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    init_sty = model.get_first_stage_encoding(
                        model.encode_first_stage(init_sty)
                    )
                    sty_z_enc, _ = sampler.encode_ddim(
                        init_sty.clone(),
                        num_steps=ddim_inversion_steps,
                        unconditional_conditioning=uc,
                        end_step=time_idx_dict[
                            ddim_inversion_steps - 1 - opt.start_step
                        ],
                        callback_ddim_timesteps=save_feature_timesteps,
                        img_callback=ddim_sampler_callback,
                    )
                    sty_feat = copy.deepcopy(feat_maps)

            with open(sty_feat_name, "wb") as h:
                pickle.dump(sty_feat, h)

        print(f"Saved features: {sty_feat_name}")

    print(f"Style feature extraction finished, elapsed: {time.time() - begin:.2f} s")


if __name__ == "__main__":
    main()
