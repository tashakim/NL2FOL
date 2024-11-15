# This executes NL2FOL model

# To run this:
# python run_nl2fol.py --model_name "llama" --nli_model_name "bert-large-mnli" --length 100


import argparse
import pandas as pd
from nl2fol import NL2FOL  # Ensure you have the NL2FOL class in `nl2fol.py`
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def setup_dataset(length=100):
    dataset = pd.read_csv('path_to_your_folio_dataset.csv').sample(length, random_state=42)
    return dataset

def main(model_name, nli_model_name, length):
    # Initialize models and tokenizers
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

    text_pipeline = pipeline(
        "text-generation", model=model_name, torch_dtype=torch.float16, max_length=1024, device_map="auto"
    )

    # Setup dataset
    dataset = setup_dataset(length=length)

    # Convert NL to FOL
    final_results = []
    for _, row in dataset.iterrows():
        nl2fol = NL2FOL(row['sentence'], 'llama', text_pipeline, tokenizer, nli_model, nli_tokenizer, debug=True)
        lf, lf2 = nl2fol.convert_to_first_order_logic()
        final_results.append({
            "sentence": row['sentence'],
            "Logical Form": lf,
            "Logical Form 2": lf2,
        })

    # Save the results
    results_df = pd.DataFrame(final_results)
    results_df.to_csv('nl2fol_results.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NL2FOL and convert text to FOL")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for text generation")
    parser.add_argument("--nli_model_name", type=str, required=True, help="Model name for NLI")
    parser.add_argument("--length", type=int, default=100, help="Number of samples to process")
    args = parser.parse_args()
    main(args.model_name, args.nli_model_name, args.length)
