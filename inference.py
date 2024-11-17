from itertools import islice
import pandas as pd
from vllm import LLM, SamplingParams
import os
import gc
from tqdm import tqdm
from PIL import Image
import ast
import config

def batch_iterator(iterable, batch_size):
    """
    Helper function to yield batches from an iterable.
    """
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch

def Inference(merged_path):
    """
    Inference function to process images and prompts in batches, generate outputs, and save results to a CSV file.
    """
    path = config.data_path

    # Load test data
    test = pd.read_csv(os.path.join(path, 'test.csv'))

    # Convert test data into a dataset for batch processing
    examples = test.to_dict(orient='records')  # Convert to list of dictionaries for iteration

    # Load category-attributes mapping
    cats = pd.read_parquet(os.path.join(path, 'category_attributes.parquet'))

    # Initialize the language model
    llm = LLM(
        merged_path,
        gpu_memory_utilization=0.80,
        trust_remote_code=True,
        max_num_seqs=10,
        dtype='half',
        max_model_len=5000
    )

    # Output file for results
    output_file = './submission.csv'

    # Ensure output columns
    output_columns = ['id', 'Category'] + [f'attr_{i+1}' for i in range(10)]

    # Initialize CSV file with headers (if file does not exist)
    if not os.path.exists(output_file):
        pd.DataFrame(columns=['id', 'Category'] + [f'attr_{i+1}' for i in range(10)]).to_csv(output_file, index=False)

    batch_size = 64  # Set batch size

    # Process in batches
    for batch in tqdm(batch_iterator(examples, batch_size)):
        prompts = []
        images = []

        # Prepare data for the current batch
        for example in batch:
            z = '0'
            s = str(example['id'])
            i = z * (6 - len(s)) + s
            image_path = f"{path}/test_images/{i}.jpg"
            image = Image.open(image_path).convert("RGB")

            category = example['Category']
            cts = cats.loc[cats['Category'] == category, 'Attribute_list'].tolist()[0]
            prompt = f"<|user|>\n<|image_1|>Retrieve Product attributes {cts} from image for the category {category}.<|end|>\n<|assistant|>\n"

            prompts.append(prompt)
            images.append(image)

        # Prepare data for the LLM
        data = [{"prompt": prompt, "multi_modal_data": {"image": image}} for prompt, image in zip(prompts, images)]

        # Generate outputs
        sampling_params = SamplingParams(temperature=0, max_tokens=256)
        outputs2 = llm.generate(data, sampling_params=sampling_params)

        # Process the outputs
        out2 = [o.outputs[0].text for o in outputs2]

        # Clean and process outputs into structured data
        df_batch = pd.DataFrame(out2, columns=['output'])
        df_batch['output_cleaned'] = df_batch['output'].apply(lambda x: str(x).replace(" {", "{").replace("} \n", "}"))
        df_batch['output_dict'] = df_batch['output_cleaned'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Extract up to 10 attributes for each output
        attributes = df_batch['output_dict'].apply(
            lambda x: list(x.values())[:10] if isinstance(x, dict) else ['dummy_value'] * 10
        )

        # Ensure exactly 10 attributes
        attributes_padded = attributes.apply(
            lambda x: x + ['dummy_value'] * (10 - len(x)) if len(x) < 10 else x
        )

        # Convert attributes into DataFrame
        df_new_batch = pd.DataFrame(attributes_padded.tolist(), columns=[f'attr_{i+1}' for i in range(10)])

        # Combine IDs, categories, and predictions
        batch_ids = pd.DataFrame(batch)[['id', 'Category']].reset_index(drop=True)
        batch_results = pd.concat([batch_ids, df_new_batch], axis=1)

        # Ensure all columns are in output_columns order
        batch_results = batch_results[output_columns]

        # Append batch results to output CSV
        batch_results.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

        # Free memory
        del prompts, images, data, outputs2, out2, df_batch, df_new_batch, batch_results
        gc.collect()

    print(f"Completed inference. Results saved to {output_file}")

if __name__ == "__main__":
    preds = Inference(config.merged_model_path)