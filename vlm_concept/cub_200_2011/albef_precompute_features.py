import argparse
import os
import logging

import numpy as np
import torch
from PIL import Image

from .cub_data import load_data
from ..ALBEF.models.tokenization_bert import BertTokenizer
from ..utils.albef_tools import Config, ALBEF, get_image_transform_fn, get_text_embeddings, get_actvs

device = "cuda" if torch.cuda.is_available() else "cpu"


def parseArguments(): 
    parser = argparse.ArgumentParser()

    # Necessary variables
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report_step", type=int, default=1)
    parser.add_argument("--text_encoder", type=str, default="bert-base-uncased")
    parser.add_argument("--data_root", type=str, default="./data/CUB_200_2011/")
    parser.add_argument("--model_path", type=str, default="./pretrained_models/albef/ALBEF.pth")
    parser.add_argument("--exp_name", type=str, default="albef_precompute_features")
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
    # Load the model and tokenizer
    logger.info("Load model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = Config()
    config = vars(config)
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=False)
    checkpoint = torch.load(args.model_path, map_location="cpu")              
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    model.to(device)
    
    block_num = 8
    model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True

    # Load image transform function
    logger.info(f"Load image transform function...")
    transform = get_image_transform_fn(config)

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
        curr_text_embeds, curr_text_masks = get_text_embeddings(model, tokenizer, text_inputs_raw)
        text_embeds_attr.append(curr_text_embeds)
        text_masks_attr.append(curr_text_masks)
    
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
        image_input = transform(image).unsqueeze(0).to(device)
        image_attributes = attributes[attributes.image_id==image_id]

        with torch.no_grad():
            # Compute image embedding for this image
            image_embeds = model.visual_encoder(image_input)
            image_masks = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)
        
        # Compute composite concept activations
        curr_actv_class = get_actvs(model, image_embeds, image_masks, text_masks_class, text_embeds_class)
        
        # Compute primitive concept activations
        # NOTE: Need gradients for forward pass in get_actvs()
        #curr_actv_attr_logits, curr_actv_attr_probs = [],[]
        #for attr_id in range(len(unique_attributes)):
        #    text_masks = text_masks_attr[attr_id]
        #    text_embeds = text_embeds_attr[attr_id]
        #    actv_attr = get_actvs(model, image_embeds, image_masks, text_embeds, text_masks)
        #    curr_actv_attr_logits.extend(actv_attr)
        #    curr_actv_attr_probs.extend(softmax(actv_attr))
        
        class_activations.append(np.array(curr_actv_class))
        #attr_activations_logits.append(np.array(curr_actv_attr_logits))
        #attr_activations_probs.append(np.array(curr_actv_attr_probs))
        
        if args.verbose and (row_id % args.report_step == 0):
            logger.info(f"{row_id}/{len(data)}")

    #attr_activations_logits = np.array(attr_activations_logits)
    #attr_activations_probs = np.array(attr_activations_probs)

    # Output results
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./outputs/cub_200_2011", exist_ok=True)
    os.makedirs(f"./outputs/cub_200_2011/{args.exp_name}", exist_ok=True)
    np.save(os.path.join(args.output_path, f"class_activations_{args.split}.npy"), class_activations)
    #np.save(os.path.join(args.output_path, f"attr_activations_logits_{args.split}.npy"), attr_activations_logits)
    #np.save(os.path.join(args.output_path, f"attr_activations_probs_{args.split}.npy"), attr_activations_probs)
    
    logger.info(f"Features output to {args.output_path}")
    logger.info("Done!")

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)

    main(args)