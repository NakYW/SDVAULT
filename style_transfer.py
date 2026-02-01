import argparse
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy
import pickle
import time
from typing import Dict, List, Optional, Tuple

import clip
import torch.nn.functional as F
import torch.optim as optim
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import torchvision.transforms as transforms

feat_maps = []

DEFAULT_CLIP_CONTENT_PROMPT_WEIGHT = 1.0
DEFAULT_CLIP_STYLE_PROMPT_WEIGHT = 1.0
DEFAULT_CLIP_CONTENT_IMAGE_WEIGHT = 1.0
DEFAULT_CLIP_STYLE_IMAGE_WEIGHT = 1.0
DEFAULT_EFDM_NOISE_STD = 0.0
DEFAULT_CONTENT_PROMPT = ""
DEFAULT_STYLE_PROMPT = ""
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_CLIP_GUIDANCE_STEPS = 0
DEFAULT_CLIP_GUIDANCE_LR = 0.05
DEFAULT_CLIP_NEGATIVE_PROMPT_WEIGHT = 0.0
DEFAULT_CLIP_MODEL = "ViT-L/14"
DEFAULT_CLIP_GUIDANCE_BLEND = 0.05


def feat_merge(opt, cnt_feats, sty_feats, start_step=0):
    feat_maps = [
        {
            "config": {
                "gamma": opt.gamma,
                "T": opt.T,
                "timestep": _,
            }
        }
        for _ in range(50)
    ]

    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue
        cnt_feat = cnt_feats[i]
        sty_feat = sty_feats[i]
        for key, value in sty_feat.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key.endswith("k") or key.endswith("v"):
                feat_maps[i][key] = value
            elif key == "z_enc":
                feat_maps[i][key] = value
        for key, value in cnt_feat.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key.endswith("q"):
                feat_maps[i][key] = value
            elif key == "z_enc":
                feat_maps[i][key] = value
    return tuple(feat_maps)


def restore_feature_precision(feature_maps, device, target_dtype=None):
    if target_dtype is None:
        target_dtype = torch.float32
    for step in feature_maps:
        for key, value in list(step.items()):
            if isinstance(value, torch.Tensor):
                step[key] = value.detach().to(
                    device=device, dtype=target_dtype, non_blocking=True
                )
    return feature_maps


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


def adain(cnt_feat, sty_feat):
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3], keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3], keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3], keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3], keepdim=True)
    output = ((cnt_feat - cnt_mean) / cnt_std) * sty_std + sty_mean
    return output


def exact_feature_distribution_matching(
    content_feat, style_feat, alpha=1.0, noise_std=0.0
):
    """
    Match per-channel feature distributions via sorting with optional relaxed strength.
    alpha controls interpolation strength (1.0 == strict EFDM).
    noise_std adds tiny Gaussian noise to alleviate rank ties.
    """
    if content_feat.shape != style_feat.shape:
        raise ValueError(
            f"EFDM expects content and style features with identical shape, got "
            f"{tuple(content_feat.shape)} vs {tuple(style_feat.shape)}"
        )

    if noise_std > 0.0:
        noise_std = float(noise_std)
        content_proc = content_feat + torch.randn_like(content_feat) * noise_std
        style_proc = style_feat + torch.randn_like(style_feat) * noise_std
    else:
        content_proc = content_feat
        style_proc = style_feat

    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))

    b, c, h, w = content_proc.shape
    content_flat = content_proc.reshape(b, c, -1)
    style_flat = style_proc.reshape(b, c, -1)

    value_content, index_content = torch.sort(content_flat, dim=-1)
    value_style, _ = torch.sort(style_flat, dim=-1)
    inverse_index = index_content.argsort(dim=-1)

    matched_style = value_style.gather(-1, inverse_index)
    delta = matched_style - content_flat.detach()
    if alpha != 1.0:
        delta = delta * alpha

    new_content = content_flat + delta
    return new_content.reshape(b, c, h, w)


def apply_init_transform(
    cnt_feat,
    sty_feat,
    mode="adain",
    efdm_alpha_pre=0.6,
    efdm_alpha_post=0.3,
    efdm_noise_std=0.0,
):
    """
    Compose AdaIN / EFDM based on selected mode.
    Modes:
        - none: return content as-is
        - adain: single AdaIN (legacy default)
        - ae: AdaIN followed by soft EFDM
        - eae: EFDM -> AdaIN -> EFDM with separate strengths
    """
    mode = mode.lower()
    if mode not in {"none", "adain", "ae", "eae"}:
        raise ValueError(f"Unsupported init transform mode: {mode}")

    if mode == "none":
        return cnt_feat

    input_dtype = cnt_feat.dtype
    if cnt_feat.dtype != torch.float32:
        cnt_proc = cnt_feat.float()
    else:
        cnt_proc = cnt_feat
    if sty_feat.dtype != cnt_proc.dtype:
        sty_proc = sty_feat.to(cnt_proc.dtype)
    else:
        sty_proc = sty_feat

    if mode == "adain":
        out = adain(cnt_proc, sty_proc)
        return out.to(input_dtype) if out.dtype != input_dtype else out

    if mode == "ae":
        adain_feat = adain(cnt_proc, sty_proc)
        out = exact_feature_distribution_matching(
            adain_feat, sty_proc, alpha=efdm_alpha_post, noise_std=efdm_noise_std
        )
        return out.to(input_dtype) if out.dtype != input_dtype else out

    # mode == "eae"
    first = exact_feature_distribution_matching(
        cnt_proc, sty_proc, alpha=efdm_alpha_pre, noise_std=efdm_noise_std
    )
    second = adain(first, sty_proc)
    out = exact_feature_distribution_matching(
        second, sty_proc, alpha=efdm_alpha_post, noise_std=efdm_noise_std
    )
    return out.to(input_dtype) if out.dtype != input_dtype else out


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


def load_prompt_table(path: Optional[str]) -> List[Dict[str, Optional[str]]]:
    table: List[Dict[str, Optional[str]]] = []
    if path is None or not os.path.isfile(path):
        return table

    def blank_entry() -> Dict[str, Optional[str]]:
        return {
            "content": None,
            "style": None,
            "content_prompt": "",
            "style_prompt": "",
            "negative_prompt": "",
        }

    buffered_entry = blank_entry()
    has_kv_data = False

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "|" in line:
                if has_kv_data:
                    table.append(buffered_entry)
                    buffered_entry = blank_entry()
                    has_kv_data = False

                parts = [part.strip() for part in line.split("|")]
                entry = blank_entry()
                if len(parts) >= 5:
                    entry["content"] = (
                        parts[0] if parts[0] and parts[0] != "*" else None
                    )
                    entry["style"] = parts[1] if parts[1] and parts[1] != "*" else None
                    entry["content_prompt"] = parts[2]
                    entry["style_prompt"] = parts[3]
                    entry["negative_prompt"] = parts[4] if parts[4] else ""
                elif len(parts) == 4:
                    entry["content"] = (
                        parts[0] if parts[0] and parts[0] != "*" else None
                    )
                    entry["style"] = parts[1] if parts[1] and parts[1] != "*" else None
                    entry["content_prompt"] = parts[2]
                    entry["style_prompt"] = parts[3]
                elif len(parts) == 3:
                    entry["content_prompt"] = parts[0]
                    entry["style_prompt"] = parts[1]
                    entry["negative_prompt"] = parts[2]
                elif len(parts) == 2:
                    entry["content_prompt"] = parts[0]
                    entry["style_prompt"] = parts[1]
                elif len(parts) == 1:
                    entry["content_prompt"] = parts[0]
                table.append(entry)
                continue

            if ":" in line:
                key, value = line.split(":", 1)
            elif "=" in line:
                key, value = line.split("=", 1)
            else:
                key, value = "content_prompt", line

            key = key.strip().lower()
            value = value.strip().strip("'\"")

            if key in {"content", "cnt"}:
                buffered_entry["content"] = None if value in {"*", ""} else value
            elif key in {"style", "sty"}:
                buffered_entry["style"] = None if value in {"*", ""} else value
            elif key in {"content_prompt", "contentprompt"}:
                buffered_entry["content_prompt"] = value
            elif key in {"style_prompt", "styleprompt"}:
                buffered_entry["style_prompt"] = value
            elif key in {"negative_prompt", "negativeprompt", "neg_prompt", "neg"}:
                buffered_entry["negative_prompt"] = value
            else:
                if buffered_entry["content_prompt"]:
                    buffered_entry["content_prompt"] += ", " + value
                else:
                    buffered_entry["content_prompt"] = value
            has_kv_data = True

    if has_kv_data:
        table.append(buffered_entry)

    return table


def resolve_prompt(
    content_name: str,
    style_name: str,
    table: List[Dict[str, Optional[str]]],
    default_content_prompt: Optional[str],
    default_style_prompt: Optional[str],
    default_negative: Optional[str],
) -> Tuple[str, str, str]:
    def normalize(text: Optional[str]) -> str:
        return text.strip() if text else ""

    selected_content = None
    selected_style = None
    selected_negative = None

    for entry in table:
        content_match = entry["content"] is None or entry["content"] == content_name
        style_match = entry["style"] is None or entry["style"] == style_name
        if content_match and style_match:
            if entry.get("content_prompt"):
                selected_content = entry["content_prompt"]
            if entry.get("style_prompt"):
                selected_style = entry["style_prompt"]
            if entry.get("negative_prompt"):
                selected_negative = entry["negative_prompt"]

    if selected_content is None:
        selected_content = default_content_prompt
    if selected_style is None:
        selected_style = default_style_prompt
    if selected_negative is None:
        selected_negative = default_negative

    return (
        normalize(selected_content),
        normalize(selected_style),
        normalize(selected_negative),
    )


def prepare_clip_image(
    image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    image = F.interpolate(
        image, size=224, mode="bicubic", align_corners=False, antialias=True
    )
    return (image - mean) / std


def compute_clip_image_features(clip_model, image_tensor: torch.Tensor) -> torch.Tensor:
    image_features = clip_model.encode_image(image_tensor)
    image_features = image_features.float()
    return image_features / image_features.norm(dim=-1, keepdim=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnt", default="./data/cnt")
    parser.add_argument("--sty", default="./data/sty")
    parser.add_argument("--ddim_inv_steps", type=int, default=50, help="DDIM eta")
    parser.add_argument("--save_feat_steps", type=int, default=50, help="DDIM eta")
    parser.add_argument("--start_step", type=int, default=49, help="DDIM eta")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta")
    parser.add_argument("--C", type=int, default=4, help="latent channels")
    parser.add_argument("--f", type=int, default=8, help="downsampling factor")
    parser.add_argument(
        "--H", 
        type=int, 
        default=512, 
        help="image height, in pixel space"
    )

    parser.add_argument(
        "--W", type=int, 
        default=512, 
        help="image width, in pixel space"
    )

    parser.add_argument(
        "--T",
        type=float,
        default=1.1,
        help="attention temperature scaling hyperparameter",
    )

    parser.add_argument(
        "--gamma", 
        type=float, 
        default=0.80, 
        help="query preservation hyperparameter"
    )

    parser.add_argument(
        "--attn_layer",
        type=str,
        default="6,7,8,9,10,11",
        help="injection attention feature layers",
    )

    parser.add_argument(
        "--model_config",
        type=str,
        default="models/ldm/v1-inference.yaml",
        help="model config",
    )

    parser.add_argument(
        "--precomputed",
        type=str,
        default="./precomputed_feats",
        help="save path for precomputed feature",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-1.5/model.ckpt",
        help="model checkpoint",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="autocast",
        help='choices: ["full", "autocast"]',
    )

    parser.add_argument(
        "--output_path", 
        type=str, 
        default="output"
    )

    parser.add_argument(
        "--without_init_adain", 
        action="store_true"
    )

    parser.add_argument(
        "--without_attn_injection", 
        action="store_true"
    )

    parser.add_argument(
        "--init_transform",
        type=str,
        default="eae",
        help="initial latent transform: none, adain, ae, or eae",
    )

    parser.add_argument(
        "--efdm_alpha_pre",
        type=float,
        default=0.35,
        help="soft EFDM strength applied before AdaIN (EAE mode)",
    )
    parser.add_argument(
        "--efdm_alpha_post",
        type=float,
        default=0.25,
        help="soft EFDM strength applied after AdaIN (AE & EAE modes)",
    )

    parser.add_argument(
        "--prompt_file",
        type=str,
        default="",
        help="optional table file: [content]|[style]|content_prompt|style_prompt|negative_prompt",
    )

    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2,
        help="classifier-free guidance scale (keep moderate so text stays auxiliary)",
    )
    opt = parser.parse_args()

    opt.efdm_noise_std = DEFAULT_EFDM_NOISE_STD
    opt.content_prompt = DEFAULT_CONTENT_PROMPT
    opt.style_prompt = DEFAULT_STYLE_PROMPT
    opt.negative_prompt = DEFAULT_NEGATIVE_PROMPT
    opt.clip_guidance_steps = DEFAULT_CLIP_GUIDANCE_STEPS
    opt.clip_guidance_lr = DEFAULT_CLIP_GUIDANCE_LR
    opt.clip_negative_prompt_weight = DEFAULT_CLIP_NEGATIVE_PROMPT_WEIGHT
    opt.clip_model = DEFAULT_CLIP_MODEL
    opt.clip_guidance_blend = DEFAULT_CLIP_GUIDANCE_BLEND
    opt.clip_content_prompt_weight = DEFAULT_CLIP_CONTENT_PROMPT_WEIGHT
    opt.clip_style_prompt_weight = DEFAULT_CLIP_STYLE_PROMPT_WEIGHT
    opt.clip_content_image_weight = DEFAULT_CLIP_CONTENT_IMAGE_WEIGHT
    opt.clip_style_image_weight = DEFAULT_CLIP_STYLE_IMAGE_WEIGHT

    if opt.H % 32 != 0 or opt.W % 32 != 0:
        raise ValueError(
            f"H/W must be multiples of 32 to avoid UNet skip-size mismatches. "
            f"Got H={opt.H}, W={opt.W}."
        )

    precision_mode = opt.precision.strip().lower()
    if precision_mode == "autocast" and torch.cuda.is_available():
        feature_dtype = torch.float16
    else:
        feature_dtype = torch.float32

    init_mode = opt.init_transform.strip().lower()
    if not init_mode:
        init_mode = parser.get_default("init_transform").strip().lower()
    allowed_init_modes = {"none", "adain", "ae", "eae"}
    if init_mode not in allowed_init_modes:
        raise ValueError(
            f"Unsupported init transform mode '{init_mode}'. "
            f"Valid options: {sorted(allowed_init_modes)}"
        )
    if opt.without_init_adain:
        init_mode = "none"
    opt.init_transform_modes = [init_mode]

    feat_path_root = opt.precomputed

    seed_everything(22)
    output_path = opt.output_path
    os.makedirs(output_path, exist_ok=True)

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
                    q_c = attn.Q_c
                    k_s = attn.K_s
                    v_s = attn.V_s
                    save_feature_map(q_c, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k_s, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v_s, f"{feature_type}_{block_idx}_self_attn_v", i)
            block_idx += 1

    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks, i, "output_block")

    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    sty_img_list = sorted(os.listdir(opt.sty))
    cnt_img_list = sorted(os.listdir(opt.cnt))

    prompt_table = load_prompt_table(opt.prompt_file)
    conditioning_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    clip_model = None
    clip_text_cache: Dict[str, torch.Tensor] = {}
    clip_mean = clip_std = None
    if opt.clip_guidance_steps > 0:
        clip_model, _ = clip.load(opt.clip_model, device=device, jit=False)
        clip_model.eval()
        clip_model.float()
        for param in clip_model.parameters():
            param.requires_grad_(False)
        clip_mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=device
        ).view(1, 3, 1, 1)
        clip_std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=device
        ).view(1, 3, 1, 1)

    def get_conditioning(
        prompt_text: str, negative_text: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (prompt_text, negative_text)
        if key not in conditioning_cache:
            positive = prompt_text if prompt_text else ""
            negative = negative_text if negative_text else ""
            c = model.get_learned_conditioning([positive])
            uc_local = model.get_learned_conditioning([negative])
            conditioning_cache[key] = (c, uc_local)
        return conditioning_cache[key]

    def get_clip_text_features(text: str) -> Optional[torch.Tensor]:
        if clip_model is None or not text:
            return None
        if text in clip_text_cache:
            return clip_text_cache[text]
        tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokens)
        text_features = text_features.float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        clip_text_cache[text] = text_features
        return text_features

    # ****************************************************************
    def run_clip_guidance(
        latent: torch.Tensor, targets: List[Tuple[str, float, torch.Tensor]]
    ) -> torch.Tensor:
        if clip_model is None or opt.clip_guidance_steps <= 0:
            return latent
        active_targets: List[Tuple[str, float, torch.Tensor]] = [
            (polarity, float(weight), feature.detach())
            for polarity, weight, feature in targets
            if feature is not None and weight > 0
        ]
        if not active_targets:
            return latent
        guide_latent = latent.detach().to(torch.float32).requires_grad_(True)
        optimizer = optim.Adam([guide_latent], lr=opt.clip_guidance_lr)
        for _ in range(opt.clip_guidance_steps):
            optimizer.zero_grad()
            decoded = model.decode_first_stage(guide_latent)
            decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
            clip_in = prepare_clip_image(decoded, clip_mean, clip_std)
            image_features = compute_clip_image_features(clip_model, clip_in)
            loss = torch.zeros([], device=guide_latent.device, dtype=torch.float32)
            for polarity, weight, feature in active_targets:
                feature = feature.to(image_features.device)
                similarity = torch.sum(image_features * feature)
                if polarity == "negative":
                    loss = loss + weight * similarity
                else:
                    loss = loss - weight * similarity
            loss.backward()
            optimizer.step()
        guided = guide_latent.detach()
        blend = float(getattr(opt, "clip_guidance_blend", 1.0))
        if blend < 1.0:
            guided = latent + blend * (guided - latent)
        return guided.to(latent.dtype)

    # ***************************************************************

    clip_content_cache: Dict[str, torch.Tensor] = {}
    clip_style_cache: Dict[str, torch.Tensor] = {}

    begin = time.time()
    for sty_name in sty_img_list:
        sty_name_ = os.path.join(opt.sty, sty_name)
        sty_base = os.path.splitext(os.path.basename(sty_name))[0]
        sty_feat_name = os.path.join(
            feat_path_root, os.path.basename(sty_name).split(".")[0] + "_sty.pkl"
        )

        if not os.path.isfile(sty_feat_name):
            print(f"Error: style feature file not found: {sty_feat_name}")
            print(
                "Please run extract_style_features.py to generate style features first."
            )
            continue

        print(f"Loading precomputed style features: {sty_feat_name}")
        with open(sty_feat_name, "rb") as h:
            sty_feat = pickle.load(h)
            sty_feat = restore_feature_precision(
                sty_feat, device, target_dtype=feature_dtype
            )
            sty_z_enc = torch.clone(sty_feat[0]["z_enc"])

        style_clip_features = None
        if clip_model is not None and opt.clip_style_image_weight > 0:
            if sty_base not in clip_style_cache:
                with torch.no_grad():
                    sty_img = load_img(sty_name_, opt.H, opt.W).to(device)
                    sty_img = torch.clamp((sty_img + 1.0) / 2.0, 0.0, 1.0)
                    clip_in = prepare_clip_image(sty_img, clip_mean, clip_std)
                    clip_style_cache[sty_base] = compute_clip_image_features(
                        clip_model, clip_in
                    )
            style_clip_features = clip_style_cache.get(sty_base)

        for cnt_name in cnt_img_list:
            cnt_name_ = os.path.join(opt.cnt, cnt_name)
            init_cnt = load_img(cnt_name_, opt.H, opt.W).to(device)

            cnt_base = os.path.splitext(os.path.basename(cnt_name))[0]

            content_prompt, style_prompt, negative_prompt = resolve_prompt(
                cnt_base,
                sty_base,
                prompt_table,
                opt.content_prompt,
                opt.style_prompt,
                opt.negative_prompt,
            )

            positive_prompt_parts = [
                part for part in (content_prompt, style_prompt) if part
            ]
            positive_prompt = ", ".join(positive_prompt_parts)

            cond, uc = get_conditioning(positive_prompt, negative_prompt)
            cond = cond.to(device)
            uc = uc.to(device)

            content_prompt_features = get_clip_text_features(content_prompt)
            style_prompt_features = get_clip_text_features(style_prompt)
            negative_prompt_features = get_clip_text_features(negative_prompt)

            content_image_features = None
            if clip_model is not None and opt.clip_content_image_weight > 0:
                if cnt_base not in clip_content_cache:
                    with torch.no_grad():
                        cnt_clip = torch.clamp((init_cnt + 1.0) / 2.0, 0.0, 1.0)
                        clip_in = prepare_clip_image(cnt_clip, clip_mean, clip_std)
                        clip_content_cache[cnt_base] = compute_clip_image_features(
                            clip_model, clip_in
                        )
                content_image_features = clip_content_cache.get(cnt_base)

            clip_targets = []
            if clip_model is not None and opt.clip_guidance_steps > 0:
                if (
                    content_prompt_features is not None
                    and opt.clip_content_prompt_weight > 0
                ):
                    clip_targets.append(
                        (
                            "positive",
                            opt.clip_content_prompt_weight,
                            content_prompt_features.detach(),
                        )
                    )
                if (
                    style_prompt_features is not None
                    and opt.clip_style_prompt_weight > 0
                ):
                    clip_targets.append(
                        (
                            "positive",
                            opt.clip_style_prompt_weight,
                            style_prompt_features.detach(),
                        )
                    )
                if (
                    content_image_features is not None
                    and opt.clip_content_image_weight > 0
                ):
                    clip_targets.append(
                        (
                            "positive",
                            opt.clip_content_image_weight,
                            content_image_features.detach(),
                        )
                    )
                if style_clip_features is not None and opt.clip_style_image_weight > 0:
                    clip_targets.append(
                        (
                            "positive",
                            opt.clip_style_image_weight,
                            style_clip_features.detach(),
                        )
                    )
                if (
                    negative_prompt_features is not None
                    and opt.clip_negative_prompt_weight > 0
                ):
                    clip_targets.append(
                        (
                            "negative",
                            opt.clip_negative_prompt_weight,
                            negative_prompt_features.detach(),
                        )
                    )

            cnt_feat_name = os.path.join(
                feat_path_root, os.path.basename(cnt_name).split(".")[0] + "_cnt.pkl"
            )
            cnt_feat = None

            if len(feat_path_root) > 0 and os.path.isfile(cnt_feat_name):
                print("Precomputed content feature loading: ", cnt_feat_name)
                with open(cnt_feat_name, "rb") as h:
                    cnt_feat = pickle.load(h)
                    cnt_feat = restore_feature_precision(
                        cnt_feat, device, target_dtype=feature_dtype
                    )
                    cnt_z_enc = torch.clone(cnt_feat[0]["z_enc"])
            else:
                latent_cnt = model.get_first_stage_encoding(
                    model.encode_first_stage(init_cnt)
                )
                cnt_z_enc, _ = sampler.encode_ddim(
                    latent_cnt.clone(),
                    num_steps=ddim_inversion_steps,
                    unconditional_conditioning=uc,
                    end_step=time_idx_dict[ddim_inversion_steps - 1 - opt.start_step],
                    callback_ddim_timesteps=save_feature_timesteps,
                    img_callback=ddim_sampler_callback,
                )
                cnt_feat = copy.deepcopy(feat_maps)
                cnt_feat = restore_feature_precision(
                    cnt_feat, device, target_dtype=feature_dtype
                )
                cnt_z_enc = cnt_feat[0]["z_enc"]

            feat_maps_template = None
            if not opt.without_attn_injection:
                feat_maps_template = feat_merge(
                    opt, cnt_feat, sty_feat, start_step=opt.start_step
                )

            for init_mode in opt.init_transform_modes:
                print(f"Style transfer: {cnt_name} + {sty_name} -> mode {init_mode}")
                if content_prompt:
                    print(f"  content_prompt: {content_prompt}")
                if style_prompt:
                    print(f"  style_prompt: {style_prompt}")
                if negative_prompt:
                    print(f"  negative_prompt: {negative_prompt}")

                init_latent = apply_init_transform(
                    cnt_z_enc,
                    sty_z_enc,
                    mode=init_mode,
                    efdm_alpha_pre=opt.efdm_alpha_pre,
                    efdm_alpha_post=opt.efdm_alpha_post,
                    efdm_noise_std=opt.efdm_noise_std,
                ).clone()

                with model.ema_scope():
                    with precision_scope("cuda"):
                        with torch.no_grad():
                            samples_ddim, _ = sampler.sample(
                                S=ddim_steps,
                                batch_size=1,
                                shape=shape,
                                verbose=False,
                                conditioning=cond,
                                unconditional_conditioning=uc,
                                eta=opt.ddim_eta,
                                x_T=init_latent,
                                injected_features=feat_maps_template,
                                start_step=opt.start_step,
                                unconditional_guidance_scale=opt.cfg_scale,
                            )

                if clip_targets:
                    samples_ddim = run_clip_guidance(samples_ddim, clip_targets)

                with model.ema_scope():
                    with torch.no_grad():
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                        )
                        x_samples_ddim = (
                            x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        )

                x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                x_sample = 255.0 * rearrange(
                    x_image_torch[0].cpu().numpy(), "c h w -> h w c"
                )
                img = Image.fromarray(x_sample.astype(np.uint8))

                output_name = f"{cnt_base}_stylized_{sty_base}_{init_mode}.png"
                img.save(os.path.join(output_path, output_name))
                print(f"Saved stylized image {os.path.join(output_path, output_name)}")

    print(f"Style transfer finished, total time: {time.time() - begin:.2f} s")


if __name__ == "__main__":
    main()
