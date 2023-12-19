import argparse
import os
import logging

import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image

from .tmn_mit_states_data import MITStatesDataset

clip_model_dict = {
    "rn50": "RN50",
    "rn101": "RN101",
    "vit_b_32": "ViT-B/32",
    "vit_l_14_336px": "ViT-L/14@336px",
    
}

device = "cuda" if torch.cuda.is_available() else "cpu"



def parseArguments(): 
    parser = argparse.ArgumentParser()

    # Necessary variables
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report_step", type=int, default=100)
    parser.add_argument("--data_root", type=str, default="./data/mit_states/")
    parser.add_argument("--model_name", type=str, choices=["rn50","rn101","vit_b_32","vit_l_14_336px"])
    args = parser.parse_args()
    
    # I/O variables
    parser.add_argument("--exp_name", type=str, default=f"clip_{args.model_name}_precompute_features")
    args = parser.parse_args()
    
    parser.add_argument("--output_path", type=str, default=f"./outputs/mit_states/{args.exp_name}")
    args = parser.parse_args()
    return args

def get_logger(args):
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./logs/mit_states", exist_ok=True)
    os.makedirs(f"./logs/mit_states/{args.exp_name}", exist_ok=True)
    logging.basicConfig(level=logging.INFO, \
            format = "%(asctime)s %(levelname)s: %(message)s", \
            datefmt = "%m/%d %H:%M:%S %p", \
            filename = f"./logs/mit_states/{args.exp_name}/{args.exp_name}.log", \
            filemode = "w"
    )
    return logging.getLogger(__name__)

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def main(args):
    # Load the model
    logger.info("Load model...")
    model, preprocess = clip.load(clip_model_dict[args.model_name], device=device)

    # Precompute features for each split
    for split in ["train", "valid", "test"]:
        # Load data
        logger.info(f"Load {split} data...")
        data = MITStatesDataset(root=args.data_root, split=split)
        logger.info(f"train pairs: {len(data.train_pairs)} | valid pairs: {len(data.valid_pairs)} | test pairs: {len(data.test_pairs)}")
        logger.info(f"train images: {len(data.train_data)} | valid images: {len(data.valid_data)} | test images: {len(data.test_data)}")

        # Prepare text inputs
        text_inputs_pair_raw = [f"this is {attr} {obj}" for attr, obj in data.pairs]
        text_inputs_attr_raw = [f"this is {attr}" for attr in data.attrs]
        text_inputs_obj_raw = [f"this is {obj}" for obj in data.objs]
        text_inputs_pair = torch.cat([clip.tokenize(text) for text in text_inputs_pair_raw]).to(device)
        text_inputs_attr = torch.cat([clip.tokenize(text) for text in text_inputs_attr_raw]).to(device)
        text_inputs_obj = torch.cat([clip.tokenize(text) for text in text_inputs_obj_raw]).to(device)

        with torch.no_grad():
            text_features_pair = model.encode_text(text_inputs_pair)
            text_features_attr = model.encode_text(text_inputs_attr)
            text_features_obj = model.encode_text(text_inputs_obj)
        text_features_pair /= text_features_pair.norm(dim=-1, keepdim=True)
        text_features_attr /= text_features_attr.norm(dim=-1, keepdim=True)
        text_features_obj /= text_features_obj.norm(dim=-1, keepdim=True)

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
        np.save(os.path.join(args.output_path, f"image_features_{split}.npy"), image_features_list)
        np.save(os.path.join(args.output_path, f"pair_activations_{split}.npy"), pair_activations)
        np.save(os.path.join(args.output_path, f"attr_activations_{split}.npy"), attr_activations)
        np.save(os.path.join(args.output_path, f"obj_activations_{split}.npy"), obj_activations)
    
    logger.info(f"Features output to {args.output_path}")
    logger.info("Done!")

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)

    main(args)

