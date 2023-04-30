from functools import partial

from PIL import Image
import torch
from torch import nn
from torchvision import transforms

from ..ALBEF.models.vit import VisionTransformer, interpolate_pos_embed
from ..ALBEF.models.xbert import BertConfig, BertForMaskedLM

device = "cuda" if torch.cuda.is_available() else "cpu"


class Config:
    def __init__(self):
        self.bert_config = "./vlm_concept/ALBEF/configs/config_bert.json"
        self.image_res = 256
        self.vision_width = 768
        self.embed_dim = 256
        self.batch_size = 64
        self.temp = 0.07
        self.mlm_probability = 0.15
        self.queue_size = 65536
        self.momentum = 0.995
        self.alpha = 0.4

class ALBEF(nn.Module):
    def __init__(
        self,                 
        text_encoder = None,
        tokenizer = None,
        config = None,    
        temp = 0.07,
        init_deit = True
    ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  
        self.itm_head = nn.Linear(text_width, 2)     

        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)       
        self.text_proj_m = nn.Linear(text_width, embed_dim)    
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        
        self.queue_ptr[0] = ptr 

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
        
def get_image_transform_fn(config):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = transforms.Compose([
        transforms.Resize((config["image_res"],config["image_res"]),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    return transform    

def get_text_embeddings(model, tokenizer, text_inputs):
    # Precompute text embeddings for primitives
    all_text_embeds, all_text_masks = [],[]
    with torch.no_grad():
        for text_input in text_inputs:
            tokenized_text = tokenizer(text_input, return_tensors="pt")
            text_ids = tokenized_text["input_ids"].to(device)
            text_masks = tokenized_text["attention_mask"].to(device)
            text_output = model.text_encoder.bert(text_ids, attention_mask=text_masks, return_dict=True, mode="text")
            text_embeds = text_output.last_hidden_state
            all_text_embeds.append(text_embeds)
            all_text_masks.append(text_masks)
    return all_text_embeds, all_text_masks

def get_actvs(model, image_embeds, image_masks, all_text_embeds, all_text_masks):
    curr_actv = []
    for text_embeds, text_masks in zip(all_text_embeds, all_text_masks):
        output = model.text_encoder.bert(
            encoder_embeds=text_embeds,
            attention_mask=text_masks,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_masks,
            return_dict=True,
            mode="fusion",
        )
        vl_embeds = output.last_hidden_state[:,0,:]
        itm_logits = model.itm_head(vl_embeds)
        curr_actv.append(itm_logits[0,1].item())
    return curr_actv
