import argparse
import os
import logging

import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image

from .tmn_mit_states_data import MITStatesDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

templates = {
    "clip": [
        "itap of a {}",
        "a bad photo of the {}",
        "a origami {}",
        "a photo of the large {}",
        "a {} in a video game",
        "art of the {}",
        "a photo of the small {}",
    ],
    "compdl": [
        "this is {}",
        "the object is {}",
        "the item is {}",
        "the item in the given picture is {}",
        "the thing in this bad photo is {}",
        "the item in the photo is {}",
        "the item in this cool photo is {}",
        "the main object in the photo is {}",
        "the item in the low resolution image is {}",
        "the object in the photo is {}",
    ],
}

def parseArguments(): 
    parser = argparse.ArgumentParser()

    # Necessary variables
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--split", type=str, choices=["train", "valid", "test"])
    parser.add_argument("--template_src", type=str, choices=["clip", "compdl"])
    parser.add_argument("--report_step", type=int, default=1)
    parser.add_argument("--data_root", type=str, default="./data/mit_states/")
    parser.add_argument("--exp_name", type=str, default="clip_prompt_precompute_features")
    args = parser.parse_args()

    # I/O variables
    parser.add_argument("--output_path", type=str, default=f"./outputs/mit_states/{args.exp_name}/{args.template_src}")
    args = parser.parse_args()
    return args

def get_logger(args):
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./logs/mit_states", exist_ok=True)
    os.makedirs(f"./logs/mit_states/{args.exp_name}", exist_ok=True)
    logging.basicConfig(level=logging.INFO, \
            format = "%(asctime)s %(levelname)s: %(message)s", \
            datefmt = "%m/%d %H:%M:%S %p", \
            filename = f"./logs/mit_states/{args.exp_name}/{args.template_src}_{args.split}.log", \
            filemode = "w"
    )
    return logging.getLogger(__name__)

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def main(args):
    # Load the model
    logger.info("Load model...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load data
    split = args.split
    logger.info(f"Load {split} data...")
    data = MITStatesDataset(root=args.data_root, split=split)
    logger.info(f"train pairs: {len(data.train_pairs)} | valid pairs: {len(data.valid_pairs)} | test pairs: {len(data.test_pairs)}")
    logger.info(f"train images: {len(data.train_data)} | valid images: {len(data.valid_data)} | test images: {len(data.test_data)}")

    # Precompute text inputs for pairs
    with torch.no_grad():
        text_features = []
        for attr, obj in data.pairs:
            texts = [template.format(" ".join([attr, obj])) for template in templates[args.template_src]]
            texts = clip.tokenize(texts).cuda()
            curr_features = model.encode_text(texts)
            curr_features /= curr_features.norm(dim=-1, keepdim=True)
            curr_features = curr_features.mean(dim=0)
            curr_features /= curr_features.norm()
            text_features.append(curr_features.detach().cpu().numpy())
        text_features_pair = np.stack(text_features)
        text_features_pair = torch.tensor(text_features_pair).cuda()
        logger.info(f"text_features_pair: {list(text_features_pair.size())}")

        # Precompute text inputs for attributes
        text_features = []
        for attr in data.attrs:
            texts = [template.format(attr) for template in templates[args.template_src]]
            texts = clip.tokenize(texts).cuda()
            curr_features = model.encode_text(texts)
            curr_features /= curr_features.norm(dim=-1, keepdim=True)
            curr_features = curr_features.mean(dim=0)
            curr_features /= curr_features.norm()
            text_features.append(curr_features.detach().cpu().numpy())
        text_features_attr = np.stack(text_features)
        text_features_attr = torch.tensor(text_features_attr).cuda()
        logger.info(f"text_features_attr: {list(text_features_attr.size())}")

        # Precompute text inputs for objects
        text_features = []
        for obj in data.objs:
            texts = [template.format(obj) for template in templates[args.template_src]]
            texts = clip.tokenize(texts).cuda()
            curr_features = model.encode_text(texts)
            curr_features /= curr_features.norm(dim=-1, keepdim=True)
            curr_features = curr_features.mean(dim=0)
            curr_features /= curr_features.norm()
            text_features.append(curr_features.detach().cpu().numpy())
        text_features_obj = np.stack(text_features)
        text_features_obj = torch.tensor(text_features_obj).cuda()
        logger.info(f"text_features_obj: {list(text_features_obj.size())}")

    # Precompute image features and pair activations
    logger.info("Start precomputing features!")
    image_features_list, pair_activations, attr_activations, obj_activations = [],[],[],[]

    for row_id, sample in enumerate(data):
        # Prepare image inputs
        image_path, attr_id, obj_id = sample["image_path"], sample["attr_id"], sample["obj_id"]
        attr, obj = sample["attr"], sample["obj"]
        image_path = os.path.join(args.data_root, "images", image_path)
        image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)

        # Take dot product of image features and text features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity_pair = (100.0 * image_features @ text_features_pair.T)
        similarity_attr = (100.0 * image_features @ text_features_attr.T)
        similarity_obj = (100.0 * image_features @ text_features_obj.T)

        image_features_list.append(image_features.detach().cpu().numpy())
        pair_activations.append(similarity_pair.detach().cpu().numpy())
        attr_activations.append(similarity_attr.detach().cpu().numpy())
        obj_activations.append(similarity_obj.detach().cpu().numpy())

        if args.verbose and (row_id % args.report_step == 0):
            logger.info(f"Progress: {row_id} / {len(data)}")

    image_features_list = np.concatenate(image_features_list)
    pair_activations = np.concatenate(pair_activations)
    attr_activations = np.concatenate(attr_activations)
    obj_activations = np.concatenate(obj_activations)

    # Output results
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./outputs/mit_states", exist_ok=True)
    os.makedirs(f"./outputs/mit_states/{args.exp_name}", exist_ok=True)
    os.makedirs(f"./outputs/mit_states/{args.exp_name}/{args.template_src}", exist_ok=True)
    os.makedirs(f"./outputs/mit_states/{args.exp_name}/{args.template_src}/combined", exist_ok=True)
    np.save(os.path.join(args.output_path, "combined", f"image_features_{split}.npy"), image_features_list)
    np.save(os.path.join(args.output_path, "combined", f"pair_activations_{split}.npy"), pair_activations)
    np.save(os.path.join(args.output_path, "combined", f"attr_activations_{split}.npy"), attr_activations)
    np.save(os.path.join(args.output_path, "combined", f"obj_activations_{split}.npy"), obj_activations)

    logger.info(f"Features output to {args.output_path}")
    logger.info("Done!")

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)

    main(args)
