import argparse
import os

from tqdm import tqdm
from transformers import DefaultDataCollator, TrainingArguments, DataCollatorWithPadding

from src.modules.data.data_utils import load_tokenizer
from src.modules.data.format_datasets import prepare_dataset_for_training
from src.modules.data.load import read_dataset_to_hf
from src.SelectiveLossTrainer import SelectiveLossTrainer
from peft import get_peft_model, LoraConfig
import torch
from omegaconf import OmegaConf

from src.modules.modeling.inference import run_inference, obtain_logit, TokenizerConfig
from src.modules.modeling.modeling_utils import setup_model, free_gpus
from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config


def main(args):
    # setting up the configs
    print("loading config file...")
    configs = load_config(args.config_file)

    # set the args to be the configs
    for key, value in args.__dict__.items():
        configs.__setattr__(key, value)

    print("executing program...")

    # we set the out_directory according to the model and dataset used
    out_directory = configs.out_directory
    os.makedirs(out_directory, exist_ok=True)

    save_config(configs, os.path.join(out_directory, "config.yaml"))

    print("output directory: ", out_directory)

    model = setup_model(configs.model_path_or_name, trust_remote_code=True)
    if model.config.max_position_embeddings < configs.max_seq_len:
        raise ValueError(f"max_seq_len {configs.max_seq_len} is greater than the model's max_position_embeddings {model.config.max_position_embeddings}")

    tokenizer = load_tokenizer(configs.tokenizer_path_or_name, configs.max_seq_len)
    print("loaded model and tokenizer! ")

    if configs.wandb.do:
        prepare_wandb(configs.wandb)

    # prepare the datasets. this is where we add the loss masks
    train_dataset, eval_datasets = prepare_dataset_for_training(configs.data_to_load,
                                                                tokenizer,
                                                                configs.seed,
                                                                configs.num_proc,
                                                                configs.max_seq_len,
                                                                configs.apply_loss_mask,
                                                                )


    assert isinstance(eval_datasets, dict), "eval_datasets should be a dictionary"

    # check if we even need to do eval
    if len(eval_datasets) == 0:
        use_eval = False
    else:
        use_eval = True

    ### setup lora
    if configs.lora.do:
        print("using lora")
        peft_config = LoraConfig(
            target_modules = list(configs.lora.lora_modules)
        )
        model = get_peft_model(model, peft_config)
        #print trainable parameters
        print("trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    ### setup the training arguments
    data_collator = DefaultDataCollator()
    training_args = TrainingArguments(
        output_dir=out_directory,
        overwrite_output_dir=True,
        eval_strategy="steps" if use_eval else "no",
        per_device_eval_batch_size=configs.eval.per_device_eval_batch_size,
        eval_steps=configs.eval.eval_steps,
        seed=configs.seed,
        report_to="wandb" if configs.wandb.do else "none",
        save_strategy="epoch" if configs.save_model else "no",
        save_total_limit=1,
        remove_unused_columns=False,
        **configs.training_args
    )

    ### setup the trainer
    trainer = SelectiveLossTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets if use_eval else None,
        tokenizer=tokenizer,
    )

    ### train the model
    trainer.train()


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

if __name__=="__main__":
    args = parse_args()
    main(args)