import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
    }
    ret.update(d)
    return ret

class Config:
    def __init__(self):
        self.exp_name = "vilt"
        self.seed = 0
        #self.datasets = ["coco", "vg", "sbu", "gcc"]
        self.datasets = []
        self.loss_names = _loss_names({"itm": 1, "mlm": 1})
        self.batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

        # Image setting
        self.train_transform_keys = ["pixelbert"]
        self.val_transform_keys = ["pixelbert"]
        self.image_size = 384
        self.max_image_len = -1
        self.patch_size = 32
        self.draw_false_image = 1
        self.image_only = False

        # Text Setting
        self.vqav2_label_size = 3129
        self.max_text_len = 40
        self.tokenizer = "bert-base-uncased"
        self.vocab_size = 30522
        self.whole_word_masking = False
        self.mlm_prob = 0.15
        self.draw_false_text = 0

        # Transformer Setting
        self.vit = "vit_base_patch32_384"
        self.hidden_size = 768
        self.num_heads = 12
        self.num_layers = 12
        self.mlp_ratio = 4
        self.drop_rate = 0.1

        # Optimizer Setting
        self.optim_type = "adamw"
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.decay_power = 1
        self.max_epoch = 100
        self.max_steps = 25000
        self.warmup_steps = 2500
        self.end_lr = 0
        self.lr_mult = 1  # multiply lr for downstream heads

        # Downstream Setting
        self.get_recall_metric = False

        # PL Trainer Setting
        self.resume_from = None
        self.fast_dev_run = False
        self.val_check_interval = 1.0
        self.test_only = True

        # below params varies with the environment
        self.data_root = ""
        self.log_dir = "result"
        self.per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
        self.num_gpus = 1
        self.num_nodes = 1
        self.load_path = "./pretrained_models/vilt/vilt_200k_mlm_itm.ckpt"
        self.num_workers = 8
        self.precision = 16

def get_cls_feats(model, text_embeds, image_embeds, text_masks, image_masks):
    co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
    co_masks = torch.cat([text_masks, image_masks], dim=1)
    x = co_embeds

    for i, blk in enumerate(model.transformer.blocks):
        x, _attn = blk(x, mask=co_masks)

    x = model.transformer.norm(x)
    cls_feats = model.pooler(x)
    return cls_feats

def get_text_embeddings(model, tokenizer, text_inputs):
    # Precompute text embeddings for all text inputs
    encoded = tokenizer(text_inputs)
    all_text_masks, all_text_embeds = [],[]
    with torch.no_grad():
        for text_id in range(len(text_inputs)):
            text_ids = torch.tensor([encoded["input_ids"][text_id]]).to(device)
            text_masks = torch.tensor([encoded["attention_mask"][text_id]]).to(device)
            text_embeds = model.text_embeddings(text_ids)
            text_embeds = text_embeds + model.token_type_embeddings(torch.zeros_like(text_masks))
            all_text_masks.append(text_masks)
            all_text_embeds.append(text_embeds)
    return all_text_masks, all_text_embeds

def get_actvs(model, image_embeds, image_masks, all_text_masks, all_text_embeds):
    curr_actv = []
    for text_masks, text_embeds in zip(all_text_masks, all_text_embeds):
        cls_feats = get_cls_feats(model, text_embeds, image_embeds, text_masks, image_masks)
        itm_logits = model.itm_score(cls_feats)
        curr_actv.append(itm_logits[0,1].item())
    return curr_actv
