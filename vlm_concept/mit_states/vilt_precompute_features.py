import argparse
import os
import logging

import numpy as np
import torch
from PIL import Image

from .tmn_mit_states_data import MITStatesDataset
from ..utils.vilt_tools import Config, get_cls_feats, get_text_embeddings, get_actvs
from ..ViLT import vilt
from ..ViLT.vilt.config import ex, config
from ..ViLT.vilt.modules import ViLTransformerSS
from ..ViLT.vilt.transforms import pixelbert_transform
from ..ViLT.vilt.datamodules.datamodule_base import get_pretrained_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def parseArguments(): 
    parser = argparse.ArgumentParser()

    # Necessary variables
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--row_start", type=int, default="0")
    parser.add_argument("--row_end", type=int, default="5000")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report_step", type=int, default=100)
    parser.add_argument("--data_root", type=str, default="./data/mit_states/")
    parser.add_argument("--exp_name", type=str, default="vilt_precompute_features")
    args = parser.parse_args()

    # I/O variables
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
            filename = f"./logs/mit_states/{args.exp_name}/{args.exp_name}_{args.split}_{args.row_start}_{args.row_end}.log", \
            filemode = "w"
    )
    return logging.getLogger(__name__)

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def main(args):
    # Load the model
    logger.info("Load model...")
    _config = Config()
    _config = vars(_config)
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
    model = ViLTransformerSS(_config)
    model.setup("test")
    model.to(device)
    model.eval()
    
    # Get data
    logger.info(f"Load {args.split} data...")
    data = MITStatesDataset(root=args.data_root, split=args.split)
    logger.info(f"train pairs: {len(data.train_pairs)} | valid pairs: {len(data.valid_pairs)} | test pairs: {len(data.test_pairs)}")
    logger.info(f"train images: {len(data.train_data)} | valid images: {len(data.valid_data)} | test images: {len(data.test_data)}")
    
    # Prepare text inputs
    logger.info(f"Prepare text inputs...")
    text_inputs_pair_raw = [f"this is {attr} {obj}" for attr, obj in data.train_pairs]
    text_inputs_attr_raw = [f"this is {attr}" for attr in data.attrs]
    text_inputs_obj_raw = [f"this is {obj}" for obj in data.objs]

    all_text_masks_pair, all_text_embeds_pair = get_text_embeddings(model, tokenizer, text_inputs_pair_raw)
    all_text_masks_attr, all_text_embeds_attr = get_text_embeddings(model, tokenizer, text_inputs_attr_raw)
    all_text_masks_obj, all_text_embeds_obj = get_text_embeddings(model, tokenizer, text_inputs_obj_raw)

    # Compute concept activations for data samples
    logger.info(f"Start computing concept activations...")
    image_features_list, pair_activations, attr_activations, obj_activations = [],[],[],[]
    for row_id, sample in enumerate(data):
        if row_id < args.row_start:
            continue
        elif row_id == args.row_end:
            break
        image_path, attr, obj = sample["image_path"], sample["attr"], sample["obj"]
        image_path = os.path.join(args.data_root, "images", image_path)
        image = Image.open(image_path).convert("RGB")
        image_input = pixelbert_transform(size=_config["image_size"])(image)
        image_input = image_input.unsqueeze(0).to(device)

        with torch.no_grad():    
            # Compute image embedding for this image
            img = image_input
            image_embeds, image_masks, patch_index, image_labels = model.transformer.visual_embed(
                img,
                max_image_len=model.hparams.config["max_image_len"],
                mask_it=False,
            )
            image_token_type_idx = 1
            image_embeds = image_embeds + model.token_type_embeddings(torch.full_like(image_masks, 1))

            # Get concept activations
            curr_actv_pair = get_actvs(model, image_embeds, image_masks, all_text_masks_pair, all_text_embeds_pair)
            curr_actv_attr = get_actvs(model, image_embeds, image_masks, all_text_masks_attr, all_text_embeds_attr)
            curr_actv_obj = get_actvs(model, image_embeds, image_masks, all_text_masks_obj, all_text_embeds_obj)

        #image_features_list.append(image_embeds.detach().cpu().numpy())
        pair_activations.append(np.array(curr_actv_pair))
        attr_activations.append(np.array(curr_actv_attr))
        obj_activations.append(np.array(curr_actv_obj))
        
        if args.verbose and (row_id % args.report_step == 0):
            logger.info(f"{row_id}/{len(data)}")
    
    #image_features_list = np.concatenate(image_features_list, axis=0)
    pair_activations = np.array(pair_activations)
    attr_activations = np.array(attr_activations)
    obj_activations = np.array(obj_activations)

    # Output results
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./outputs/mit_states", exist_ok=True)
    os.makedirs(f"./outputs/mit_states/{args.exp_name}", exist_ok=True)
    #np.save(os.path.join(args.output_path, f"image_features_{args.split}.npy"), image_features_list)
    np.save(os.path.join(args.output_path, f"pair_activations_{args.split}_{args.row_start}_{args.row_end}.npy"), pair_activations)
    np.save(os.path.join(args.output_path, f"attr_activations_{args.split}_{args.row_start}_{args.row_end}.npy"), attr_activations)
    np.save(os.path.join(args.output_path, f"obj_activations_{args.split}_{args.row_start}_{args.row_end}.npy"), obj_activations)
    
    logger.info(f"Features output to {args.output_path}")
    logger.info("Done!")

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)

    main(args)
