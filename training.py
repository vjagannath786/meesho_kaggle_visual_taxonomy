
import os



import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments
from transformers import BitsAndBytesConfig
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset
from accelerate import Accelerator
from peft import LoraConfig, PeftModel
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import f1_score
import subprocess
import ast

torch.manual_seed(2024)





path = '/home/visual-taxonomy'

def create_dataset():
    

    train = pd.read_csv(os.path.join(path, 'train.csv'))

    test = pd.read_csv(os.path.join(path, 'test.csv'))

    cats = pd.read_parquet(os.path.join(path, 'category_attributes.parquet'))
    
    #### Fill NaN with dummy_value 
    train.loc[(train['Category'] == 'Men Tshirts') & (train['attr_1'].notna()) & (train['attr_2'].notna())& (train['attr_3'].notna())
    & (train['attr_4'].notna()) & (train['attr_5'].notna()), ['attr_6','attr_7','attr_8','attr_9','attr_10']] = 'dummy_value'


    train.loc[train['Category'] == 'Kurtis', 'attr_10'] = 'dummy_value'

    train.loc[train['Category'] == 'Women Tshirts', ['attr_9','attr_10']] = 'dummy_value'

    #train = train.fillna('unknown')


    ### Final dataset to train
    df = train.dropna()

    #df_new = pd.read_csv('/dbfs/FileStore/filled_sarees.csv')

    #df = pd.concat([df_old, df_new], axis=0)

    #print(df.shape)


    
    k = 5  # or any other number of folds you want

    # Initialize the StratifiedKFold object
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Define your target column for stratification
    target_column = 'Category'  # replace with the name of your target column

    # Create a list of train and validation DataFrames for each fold
    folds = []
    for train_index, val_index in skf.split(df, df[target_column]):
    
        folds.append((train_index, val_index))
        
        
    trn_idx, val_idx = folds[0]

    train_df = df.iloc[trn_idx]
    val_df = df.iloc[val_idx]


    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(train_df.shape, val_df.shape)


    

    dataset = Dataset.from_pandas(train_df)
    
    eval_dataset = Dataset.from_pandas(val_df)
    
    
    return dataset, eval_dataset
        
    














class DataCollator:
    def __init__(self, processor):
        self.processor = processor

    

    def __call__(self, examples):
        
        IGNORE_INDEX = -100

        all_input_ids = []
        all_label_ids = []
        all_pixel_values = []
        all_image_sizes = []

        for example in examples:
            
        
            z ='0'

            s= str(example['id'])
            
            i = z*(6-len(s)) + s
        
            image_path = f"/home/visual-taxonomy/train_images/{i}.jpg"
            path = '/home/visual-taxonomy'
        
            category = example['Category']
        
            cats = pd.read_parquet(os.path.join(path, 'category_attributes.parquet'))
        
            cts = cats.loc[cats['Category'] == category, 'Attribute_list'].tolist()[0]

            d = {}

            for i in range(len(cts)):
                d[cts[i]] = example[f'attr_{i+1}']
        
            image = Image.open(image_path).convert("RGB")
            #text_dict = example['texts'][0]

            question = f"""Retrieve Product attributes {cts} from image for the category {category}."""
        
            answer = d
            prompt_message = {
            'role': 'user',
            'content': f'<|image_1|>\n{question}',
            }
        
            #print(prompt_message)

            prompt = self.processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
            )
            answer = f'{answer}<|end|>\n<|endoftext|>'

            # mask questions for labels
            batch = self.processor(prompt, [image], return_tensors='pt')
            prompt_input_ids = batch['input_ids']
            # Do not add bos token to answer
            answer_input_ids = self.processor.tokenizer(
                answer, add_special_tokens=False, return_tensors='pt'
            )['input_ids']
            input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)
            ignore_index = -100
            # labels = torch.cat(
            #     [
            #         torch.tensor([ignore_index] * len(prompt_input_ids[0])).unsqueeze(0),
            #         answer_input_ids,
            #     ],
            #     dim=1,
            # )
            labels = torch.cat([torch.full((1, len(prompt_input_ids[0])), IGNORE_INDEX), 
                                answer_input_ids], dim=1)

            # batch['input_ids'] = input_ids
            # del batch['attention_mask']
            # batch['labels'] = labels

            # prepare expected shape for pad_sequence
            all_input_ids.append(input_ids.squeeze(0))
            all_label_ids.append(labels.squeeze(0))

            all_pixel_values.append(batch['pixel_values'])
            all_image_sizes.append(batch['image_sizes'])
        

    
        input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
        labels = pad_sequence(all_label_ids, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
        pixel_values = torch.cat(all_pixel_values, dim=0)
        image_sizes = torch.cat(all_image_sizes, dim=0)

        inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'pixel_values': pixel_values,
        'image_sizes': image_sizes,
        }
        
        
        return inputs
    
def create_lora_config(rank, alpha_to_rank_ratio=2.0, dropout=0.0, freeze_vision_model=False):
    linear_modules = [
        # Phi language modules
        'qkv_proj',  # attention
        'o_proj',
        'down_proj',  # MLP
        'gate_up_proj',
        'lm_head',
    ]
    if not freeze_vision_model:
        vision_linear_modules = [
            # CLIP modules
            'q_proj',  # attention
            'k_proj',
            'v_proj',
            'out_proj',
            'fc1',  # MLP
            'fc2',
            # image projection
            'img_projection.0',
            'img_projection.2',
        ]
        linear_modules.extend(vision_linear_modules)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=round(rank * alpha_to_rank_ratio),
        lora_dropout=dropout,
        target_modules=linear_modules,
        init_lora_weights='gaussian',
    )
    return lora_config

def create_model(model_name_or_path, use_flash_attention=False, use_qlora=False):
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        )
        if use_qlora
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        # Phi-3-V is originally trained in bf16 + flash attn
        # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float16,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'eager'
        #, quantization_config=bnb_config
    )

    return model

def patch_clip_for_lora(model):
    # remove unused parameters and then monkey patch
    def get_img_features(self, img_embeds):
        clip_vision_model = self.img_processor.vision_model
        hidden_states = clip_vision_model.embeddings(img_embeds)
        hidden_states = clip_vision_model.pre_layrnorm(hidden_states)
        patch_feature = clip_vision_model.encoder(
            inputs_embeds=hidden_states, output_hidden_states=True
        ).hidden_states[-1][:, 1:]
        return patch_feature

    image_embedder = model.model.vision_embed_tokens
    layer_index = image_embedder.layer_idx
    clip_layers = image_embedder.img_processor.vision_model.encoder.layers
    if layer_index < 0:
        layer_index = len(clip_layers) + layer_index
    del clip_layers[layer_index + 1 :]
    del image_embedder.img_processor.vision_model.post_layernorm
    image_embedder.get_img_features = get_img_features.__get__(image_embedder)
    


def attribute_f1_score(micro_f1, macro_f1):
    if micro_f1 + macro_f1 == 0:
        return 0
    return 2 * (micro_f1 * macro_f1) / (micro_f1 + macro_f1)



def get_score(cats, val_df, preds):

    attribute_score = {}

    total_score = []


    for cat, val, l in zip(cats.Category.unique(), cats.No_of_attribute, cats.Attribute_list):

        print(cat)

        attribute_score.setdefault(cat, [])

        for i in range(1, val+1):

            print(f'for {cat} and column {l[i-1]}')

            macro = f1_score(val_df.loc[val_df['Category'] == cat][f'attr_{i}'], preds.loc[preds['Category'] == cat][f'attr_{i}'], average='macro')
            micro = f1_score(val_df.loc[val_df['Category'] == cat][f'attr_{i}'], preds.loc[preds['Category'] == cat][f'attr_{i}'], average='micro')

            score = attribute_f1_score(micro, macro)

            print(f'score is {score}')

            attribute_score[cat].append(score)

        

        total_score.append(sum(attribute_score[cat]) / len(attribute_score[cat]))


    
    print(f'total scores is {total_score}')

    final_score = sum(total_score) / len(total_score)

    print(f'final score is {final_score}')

    return final_score



def install_vllm():
    

    subprocess.run(['pip', 'install', 'vllm==0.6.1.post1'])

    #subprocess.run(['pip', 'install', 'flash-attn==2.6.3'])

    #subprocess.run(['pip', 'install', '--upgrade','typing_extensions'])



def evaluate(merged_path, eval_dataset):

    #model_id = 'microsoft/Phi-3.5-vision-instruct'
    #merged_path = '/home/ml/merged_v1'

    #install_vllm()


    from vllm import LLM, SamplingParams

    


    llm = LLM(merged_path, gpu_memory_utilization=0.80, trust_remote_code=True, max_num_seqs=10, max_model_len=37106)


    # Prepare lists to batch the prompts and images
    prompts = []
    images = []



    examples = eval_dataset



    cats = pd.read_parquet(os.path.join(path, 'category_attributes.parquet'))
        
    #cts = cats.loc[cats['Category'] == category, 'Attribute_list'].tolist()[0]

    for example in tqdm(examples):
       
        # Prepare image and prompt as before
        #print(example)
        z = '0'
        s = str(example['id'])
        i = z * (6 - len(s)) + s
        image_path = f"/home/visual-taxonomy/train_images/{i}.jpg"
        image = Image.open(image_path).convert("RGB")
    
        category = example['Category']
        cts = cats.loc[cats['Category'] == category, 'Attribute_list'].tolist()[0]
        prompt = f"<|user|>\n<|image_1|>Retrieve Product attributes {cts} from image for the category {category}.<|end|>\n<|assistant|>\n"
    
        # Append prompt and image to lists
        prompts.append(prompt)
        images.append(image)

    
    data = [{"prompt": prompt, "multi_modal_data": {"image": image}} for prompt, image in zip(prompts, images)]


    sampling_params = SamplingParams(temperature=0, max_tokens=256)


    outputs2 = llm.generate(data, sampling_params=sampling_params)


    out2 = []
    
    for o in tqdm(outputs2):
        generated_text = o.outputs[0].text
        out2.append(generated_text)

    

    df2 = pd.DataFrame(out2)


    df2.to_csv('./jsonoutput.csv')

    df2[1] = df2[0].apply(lambda x: str(x).replace(" {","{").replace("} \n","}"))

    df2[2] = df2[1].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    df_new_2 = pd.DataFrame(df2[2].apply(lambda x: list(x.values())).tolist(), columns=[f'attr_{i+1}' for i in range(10)]).fillna('dummy_value')

    eval_df = eval_dataset.to_pandas()


    preds = pd.concat([eval_df[['id','Category']].reset_index(drop=True), df_new_2], axis=1)


    final_score = get_score(cats, eval_df, preds)



    return final_score






    





    
def main():



    

    

    



    args = {
    'model_name_or_path': 'microsoft/Phi-3.5-vision-instruct',
    'use_flash_attention': True,
    'bf16': True,
    'use_lora': True,
    'use_qlora': False,
    'output_dir': '/home/ml/output_A100_batch2/',
    'batch_size': 16,
    'num_crops': 16,
    'num_train_epochs': 1,
    'learning_rate': 3e-5,
    'wd': 0.01,
    'tqdm': True,
    'lora_rank': 16,
    'lora_alpha_ratio': 2,
    'lora_dropout': 0.0,
    'freeze_vision_model': False
}
    
    assert args['num_crops'] <= 16, 'num_crops must be less than or equal to 16'
    if args['use_qlora']:
        args['use_lora'] = True

    accelerator = Accelerator()
    
    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            args['model_name_or_path'], trust_remote_code=True, num_crops=args['num_crops']
        )

        

        model = create_model(
            args['model_name_or_path'],
            use_flash_attention=args['use_flash_attention'],
            use_qlora=args['use_qlora'],
        )
    
    
    train_dataset, eval_dataset = create_dataset()
    
    num_gpus = accelerator.num_processes
    print(f'training on {num_gpus} GPUs')
    assert args['batch_size'] % num_gpus == 0, 'Batch size must be divisible by the number of GPUs'
    gradient_accumulation_steps = args['batch_size'] // num_gpus
    if args['bf16']:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False
        
    
    
    # hard coded training args
    training_args = TrainingArguments(
        num_train_epochs=args['num_train_epochs'],
        #max_steps = 500,
        per_device_train_batch_size=2,  
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},  # NOTE important for LoRA
        gradient_accumulation_steps=4,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args['learning_rate'],
        weight_decay=args['wd'],
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=50,
        logging_steps=10,
        output_dir=args['output_dir'],
        save_strategy='no',
        save_total_limit=10,
        save_only_model=True,
        bf16=True,
        fp16=False,
        remove_unused_columns=False,
        report_to='none',
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        ddp_find_unused_parameters=False,
    )
    
    data_collator = DataCollator(processor)
    
    
    if not args['use_qlora']:
        print('model to cuda')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        model = model.to(f'cuda:{local_rank}')
        
    if args['use_lora']:
        print('using lora')
        patch_clip_for_lora(model)
        lora_config = create_lora_config(
            rank=args['lora_rank'],
            alpha_to_rank_ratio=args['lora_alpha_ratio'],
            dropout=args['lora_dropout'],
            freeze_vision_model=args['freeze_vision_model'],
        )
        model.add_adapter(lora_config)
        model.enable_adapters()
        
        
    if args['freeze_vision_model']:
        model.model.vision_embed_tokens.requires_grad_(False)

    
    
        
    
    ### Important fix to avoid 
    ##ValueError: Attempting to unscale FP16 gradients
    for param in model.parameters():
        
        if param.requires_grad:
            param.data = param.data.float()
        
    

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
        )
    
    trainer.train()
    
    trainer.save_model()

    
    if accelerator.is_main_process:
        processor.tokenizer.save_pretrained(training_args.output_dir)

    
    print('save completed')

    merged_path = '/home/ml/merged_v1'

    base_model = create_model(
            args['model_name_or_path'],
            use_flash_attention=args['use_flash_attention'],
            use_qlora=args['use_qlora'],
        )


    peft_model = PeftModel.from_pretrained(
     base_model, args['output_dir']
    )

    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(merged_path, safe_serialization=False)

    try:
        processor.save_pretrained(merged_path)
    except:
        processor.tokenizer.save_pretrained(merged_path)


    return merged_path, eval_dataset 


if __name__ == "__main__":

    #_, eval_dataset = create_dataset()
    ####

    output, val_df = main()
    #import torch.multiprocessing as mp

    #mp.set_start_method('spawn')

    score = evaluate('/home/ml/merged_v1', val_df)

    print(score)