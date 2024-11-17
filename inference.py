import pandas as pd
from vllm import LLM, SamplingParams
import os
import gc
import datasets
from tqdm import tqdm
from PIL import Image
import ast

###


def Inference(merged_path):


    path = '/home/visual-taxonomy'

    test = pd.read_csv(os.path.join(path, 'test.csv'))



    eval_dataset = datasets.Dataset.from_pandas(test)


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
        image_path = f"/home/visual-taxonomy/test_images/{i}.jpg"
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

    preds.to_csv('submission.csv', index=False)


    return preds



if __name__ == "__main__":

    preds = Inference('./ml/merged_v1')