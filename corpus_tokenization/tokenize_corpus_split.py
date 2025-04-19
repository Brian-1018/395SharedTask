# takes in the input directory, output directory, path to the tokenizer, and the max sequence length
# the input directory is the directory containing N sharded jsonl files
# the output directory is the directory where the each file is tokenized

# Edit 4/10/25: Add in split to create validation set

from tokenizers import Tokenizer
import json
import argparse
import torch
from tqdm import tqdm
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=Path, default="../data")
    parser.add_argument("--train_file", type=Path, default="train_10M.jsonl")
    parser.add_argument("--valid_file", type=Path, default=None)
    parser.add_argument("--tokenizer_folder", type=Path, default="../tokenizers")
    parser.add_argument("--tokenizer_file", type=Path, default="tokenizer_10M.json")
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def tokenize_text(tokenizer, text):
    # text = text.strip() # NOT WORKING, 
    text = text["text"].strip()
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    ids = torch.tensor(ids, dtype=torch.int16)
    return ids


def tokenize_file(input_filename, output_filename, tokenizer):
    tokenized_documents = []
    n_subwords = 0

    for i, line in enumerate(tqdm(input_filename.open('rt'))):
        document = json.loads(line)
        tokenized_document = tokenize_text(tokenizer, document)
        tokenized_documents.append(tokenized_document)
        n_subwords += len(tokenized_document)

        if i == 0:
            print("Example tokenized document:")
            print(document)
            for token in tokenized_document:
                print(tokenizer.id_to_token(token), end=" ")
            print(flush=True)

    torch.save(tokenized_documents, output_filename)
    print(f"Tokenized {len(tokenized_documents)} documents with {n_subwords} subwords in total")
    return tokenized_documents


if __name__ == "__main__":
    args = parse_args()

    name = f"_{args.name}" if args.name is not None else ""

    tokenizer_path = args.tokenizer_folder / args.tokenizer_file
    input_train_path = args.data_folder / args.train_file

    # load the tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # If no separate validation file is provided, tokenize the train file and split into train, validation, and test sets.
    if args.valid_file is None:
        # Tokenize the entire training file and get the list of tokenized documents.
        tokenized_train = tokenize_file(
            input_train_path, 
            input_train_path.with_name(f"{input_train_path.stem}{name}_tokenized.bin"), 
            tokenizer
        )
        
        import random
        random.seed(42)
        random.shuffle(tokenized_train)
        
        n_total = len(tokenized_train)
        valid_ratio = 0.1
        test_ratio = 0.1
        n_valid = int(n_total * valid_ratio)
        n_test = int(n_total * test_ratio)
        n_train = n_total - n_valid - n_test
        
        train_split = tokenized_train[:n_train]
        valid_split = tokenized_train[n_train:n_train+n_valid]
        test_split = tokenized_train[n_train+n_valid:]
        
        output_train_split_path = input_train_path.with_name(f"{input_train_path.stem}{name}_split_train_tokenized.bin")
        output_valid_path = input_train_path.with_name(f"{input_train_path.stem}{name}_valid_tokenized.bin")
        output_test_path = input_train_path.with_name(f"{input_train_path.stem}{name}_test_tokenized.bin")
        
        torch.save(train_split, output_train_split_path)
        torch.save(valid_split, output_valid_path)
        torch.save(test_split, output_test_path)
        
        print(f"Split tokenized data into {len(train_split)} train, {len(valid_split)} validation, and {len(test_split)} test documents.")
    else:
        # If a separate validation file is provided, tokenize both the training and validation files separately.
        tokenize_file(
            input_train_path, 
            input_train_path.with_name(f"{input_train_path.stem}{name}_tokenized.bin"), 
            tokenizer
        )
        input_valid_path = args.data_folder / args.valid_file
        output_valid_path = input_valid_path.with_name(f"{input_valid_path.stem}{name}_tokenized.bin")
        tokenize_file(input_valid_path, output_valid_path, tokenizer)
