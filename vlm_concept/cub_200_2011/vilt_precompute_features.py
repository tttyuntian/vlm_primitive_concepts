import argparse
import os
import logging

import numpy as np
import torch
from PIL import Image

from .cub_data import load_data
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
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report_step", type=int, default=100)
    parser.add_argument("--data_root", type=str, default="./data/CUB_200_2011/")
    parser.add_argument("--exp_name", type=str, default="vilt_precompute_features")
    args = parser.parse_args()

    # I/O variables
    parser.add_argument("--output_path", type=str, default=f"./outputs/cub_200_2011/{args.exp_name}")
    args = parser.parse_args()
    return args

def get_logger(args):
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./logs/cub_200_2011", exist_ok=True)
    os.makedirs(f"./logs/cub_200_2011/{args.exp_name}", exist_ok=True)
    logging.basicConfig(level=logging.INFO, \
            format = "%(asctime)s %(levelname)s: %(message)s", \
            datefmt = "%m/%d %H:%M:%S %p", \
            filename = f"./logs/cub_200_2011/{args.exp_name}/{args.exp_name}_{args.split}.log", \
            filemode = "w"
    )
    return logging.getLogger(__name__)

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def softmax(inputs):
    res = torch.tensor(inputs).float()
    res = res.softmax(dim=-1)
    return res.numpy()
        
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
    data, attributes, unique_attributes = load_data(args, split=args.split)
    logger.info(f"Data size: {len(data)}")
    
    # Precompute text embeddings for primitives
    logger.info(f"Prepare text inputs...")
    text_masks_attr, text_embeds_attr = [],[]
    sample_image_attributes = attributes[attributes.image_id==1]
    for uni_attr in unique_attributes:
        curr_attr_df = sample_image_attributes[sample_image_attributes.attribute_template==uni_attr]
        text_inputs_raw = [
            "a photo of bird whose {} is {}".format(attr.replace("has ", ""), label)
            for attr, label in zip(curr_attr_df["attribute_template"], curr_attr_df["attribute_label"])
        ]
        curr_text_masks_attr, curr_text_embeds_attr = get_text_embeddings(model, tokenizer, text_inputs_raw)
        text_masks_attr.append(curr_text_masks_attr)
        text_embeds_attr.append(curr_text_embeds_attr)

    # Precompute text embeddings for composites
    text_inputs_raw = [f"the bird is {class_name}" for class_name in data.class_name.unique()]
    text_masks_class, text_embeds_class = get_text_embeddings(model, tokenizer, text_inputs_raw)

    # Compute concept activations for data samples
    logger.info(f"Start computing concept activations...")
    image_features_list, class_activations, attr_activations_logits, attr_activations_probs = [],[],[],[]
    for row_id in range(len(data)):
        # Prepare image inputs
        image_id, class_id, image_name, is_training = data.iloc[row_id][["image_id", "class_id", "filepath", "is_training_image"]]
        image_path = os.path.join(args.data_root, "CUB_200_2011", "images", image_name)
        image = Image.open(image_path).convert("RGB")
        image_input = pixelbert_transform(size=_config["image_size"])(image)
        image_input = image_input.unsqueeze(0).to(device)
        image_attributes = attributes[attributes.image_id==image_id]

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

            # Compute composite concept activations
            curr_actv_class = get_actvs(model, image_embeds, image_masks, text_masks_class, text_embeds_class)

            # Compute primitive concept activations
            curr_actv_attr_logits, curr_actv_attr_probs  = [],[]
            for attr_id in range(len(unique_attributes)):
                text_masks = text_masks_attr[attr_id]
                text_embeds = text_embeds_attr[attr_id]
                actv_attr = get_actvs(model, image_embeds, image_masks, text_masks, text_embeds)
                curr_actv_attr_logits.extend(actv_attr)
                curr_actv_attr_probs.extend(softmax(actv_attr))

        #image_features_list.append(image_embeds.detach().cpu().numpy())
        class_activations.append(np.array(curr_actv_class))
        attr_activations_logits.append(np.array(curr_actv_attr_logits))
        attr_activations_probs.append(np.array(curr_actv_attr_probs))
        
        if args.verbose and (row_id % args.report_step == 0):
            logger.info(f"{row_id}/{len(data)}")
        
    class_activations = np.array(class_activations)
    attr_activations_logits = np.array(attr_activations_logits)
    attr_activations_probs = np.array(attr_activations_probs)

    # Output results
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./outputs/cub_200_2011", exist_ok=True)
    os.makedirs(f"./outputs/cub_200_2011/{args.exp_name}", exist_ok=True)
    #with open(os.path.join(args.output_path, f"image_features_{args.split}.pkl"), "wb") as f:
    #    pkl.dump(image_features_list, f)
    np.save(os.path.join(args.output_path, f"class_activations_{args.split}.npy"), class_activations)
    np.save(os.path.join(args.output_path, f"attr_activations_logits_{args.split}.npy"), attr_activations_logits)
    np.save(os.path.join(args.output_path, f"attr_activations_probs_{args.split}.npy"), attr_activations_probs)
    
    logger.info(f"Features output to {args.output_path}")
    logger.info("Done!")

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)

    main(args)