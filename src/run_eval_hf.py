import argparse
import json
import os

from tqdm import tqdm
from datasets import load_dataset
from src.modules.data.data_utils import load_tokenizer

from src.modules.modeling.modeling_utils import setup_model
from src.modules.templates import TOFU_QUERY_TEMPLATE
from src.modules.utils import load_config, save_config


def eval_tofu(model, tokenizer, out_directory, configs):
    """Evaluates on tofu using custom metrics"""

    # load the dataset and select the necessary ones
    dataset = load_dataset("locuslab/TOFU", name="retain_perturbed")["train"]

    # reformat the dataset such that it is in generation format
    def reformat_row(row, format):
        question = row["question"]

        prompt = format.format(question=question)

        return {"prompt": prompt}

    format = TOFU_QUERY_TEMPLATE
    dataset = dataset.map(reformat_row, fn_kwargs={"format": format}, batched=False)

    # runs the generation and saves the output
    out_fn = os.path.join(out_directory, "generation_output.jsonl")
    print("saving to ", out_fn)

    # in generation tasks, we pad from the left side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    with open(out_fn, "w") as out_file:
        progress_bar = tqdm(total=len(dataset))

        ind = 0
        while ind < len(dataset):
            prompts = dataset[ind:ind + configs.batch_size]["prompt"]

            model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            generated_ids = model.generate(**model_inputs, **configs.generation_kwargs)
            final = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for i in range(len(prompts)):
                out_file.write(json.dumps({"completion": final[i][len(prompts[i]):],
                                           "prompt": prompts[i]}) + "\n")

            ind += configs.batch_size
            progress_bar.update(configs.batch_size)

    progress_bar.close()

def main(args):
    # setting up the configs
    print("loading config file...")
    configs = load_config(args.config_file)

    # set the args to be the configs
    for key, value in args.__dict__.items():
        configs.__setattr__(key, value)

    print("executing command...")

    out_directory = configs.out_directory
    os.makedirs(out_directory, exist_ok=True)

    # saves the config
    save_config(configs, os.path.join(out_directory, "config.yaml"))

    model = setup_model(configs.model_path_or_name)

    tokenizer = load_tokenizer(configs.tokenizer_path_or_name)

    if configs.exp_name == "tofu":
        eval_tofu(model, tokenizer, out_directory, configs)

    print("yay!")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="(input) type of dataset we're creating"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="(input) the path to the config file"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)