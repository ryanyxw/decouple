# this file is responsible for reformatting the datasets into a unified format
import re

from datasets import concatenate_datasets, Dataset

from src.modules.data.format_utils import reformat_dialogue_with_template, select_binary_balanced_dataset, \
    partition_dataset, preprocess_conversation, tokenize_input_output_pair

from src.modules.data.load import read_dataset_to_hf
from src.modules.templates import *





def prepare_wildguard_prompt_dataset(tokenizer, seed, max_seq_len, num_proc=1, use_loss_mask=False):
    # we first read wildguard dataset from huggingface
    train_dataset = read_dataset_to_hf("allenai/wildguardmix", name="wildguardtrain")["train"].shuffle(seed=seed)
    test_dataset = read_dataset_to_hf("allenai/wildguardmix", name="wildguardtest")["test"].shuffle(seed=seed)

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        final_instruction = prompt.format(human_request=row["prompt"])
        is_prompt_harmful = row["prompt_harm_label"] == 'harmful'
        if (add_label):
            final_instruction += WILDGUARD_PROMPT_ONLY_LABELS[is_prompt_harmful]

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "is_adversarial": False,
                    "is_harmful": is_prompt_harmful,
                    "skip": True}

        attention_mask = [1] * len(input_ids)
        loss_mask = [0] * (len(input_ids) - 1) + [1]

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "is_adversarial": row["adversarial"],
                "is_harmful": is_prompt_harmful,
                "skip": False}

    prompt = WILDGUARD_PROMPT_ONLY_TEMPLATE

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True}, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False}, remove_columns=test_dataset.column_names)

    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    return train_dataset, test_dataset


def prepare_wildguard_undesired_prompt_dataset(tokenizer, seed, max_seq_len, num_proc=1, use_loss_mask=False):
    """ this experiment first filters the wildguard dataset for desired cateogires, and then evaluates model capability at predicting if response is harmful or not"""

    # we first read wildguard dataset from huggingface
    train_dataset = read_dataset_to_hf("allenai/wildguardmix", name="wildguardtrain")["train"].shuffle(seed=seed)
    test_dataset = read_dataset_to_hf("allenai/wildguardmix", name="wildguardtest")["test"].shuffle(seed=seed)

    # we first filter the dataset for the subcategories we want to use
    target_categories = ["social_stereotypes_and_unfair_discrimination", "toxic_language_hate_speech", "defamation_encouraging_unethical_or_unsafe_actions"]

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        final_instruction = prompt.format(human_request=row["prompt"])
        is_prompt_harmful = row["prompt_harm_label"] == 'harmful'
        if (add_label):
            final_instruction += WILDGUARD_PROMPT_ONLY_LABELS[is_prompt_harmful]

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "is_adversarial": False,
                    "is_harmful": is_prompt_harmful,
                    "skip": True}

        attention_mask = [1] * len(input_ids)
        loss_mask = [0] * (len(input_ids) - 1) + [1]

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "is_adversarial": row["adversarial"],
                "is_harmful": is_prompt_harmful,
                "skip": False}

    prompt = WILDGUARD_PROMPT_ONLY_TEMPLATE

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True}, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False}, remove_columns=test_dataset.column_names)

    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    return train_dataset, test_dataset

def prepare_paradetox_dataset(tokenizer, seed, max_seq_len, num_proc):
    # we first read wildguard dataset from huggingface
    train_dataset = read_dataset_to_hf("s-nlp/paradetox")["train"].shuffle(seed=seed)

    # create the dataset with 90-10 split between train and test
    test_dataset = train_dataset.select(range(int(0.1 * len(train_dataset))))
    train_dataset = train_dataset.select(range(len(test_dataset), len(train_dataset)))

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        final_instruction = prompt.format(toxic_text=row["en_toxic_comment"])
        if (add_label):
            final_instruction += row["en_neutral_comment"] + tokenizer.eos_token

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "golden_label": row["en_neutral_comment"],
                    "skip": True}

        # add masks to the input
        if (add_label):
            question = tokenizer.encode(prompt.format(toxic_text=row["en_toxic_comment"]))
            answer = tokenizer.encode(row["en_neutral_comment"]) + [tokenizer.eos_token_id]
            temp_question = question + answer

            if temp_question != input_ids[:len(temp_question)]:
                import pdb
                pdb.set_trace()
            attention_mask = [1] * len(input_ids)
            loss_mask = [0] * (len(input_ids) - len(answer)) + [1] * len(answer)
        else:
            attention_mask = [1] * len(input_ids)
            loss_mask = [0] * len(input_ids)

        # we prepare the golden label, which must also be tokenized and padded
        golden_label = tokenizer.encode(row["en_neutral_comment"], padding="max_length", max_length=max_seq_len - 1)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "golden_label": golden_label,
                "skip": False}

    prompt = PARADETOX_PROMPT_ONLY_TEMPLATE

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=1,
                                      fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True},
                                      remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=1,
                                    fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False},
                                    remove_columns=test_dataset.column_names)


    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    return train_dataset, test_dataset


def prepare_paradetox_lowdata_dataset(tokenizer, seed, max_seq_len, num_proc):
    # we first read wildguard dataset from huggingface
    train_dataset = read_dataset_to_hf("s-nlp/paradetox")["train"].shuffle(seed=seed)

    num_train = 3200
    # create the dataset with 90-10 split between train and test
    test_dataset = train_dataset.select(range(int(0.1 * len(train_dataset))))
    train_dataset = train_dataset.select(range(len(test_dataset), len(test_dataset) + num_train))

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        final_instruction = prompt.format(toxic_text=row["en_toxic_comment"])
        if (add_label):
            final_instruction += row["en_neutral_comment"] + tokenizer.eos_token

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "golden_label": row["en_neutral_comment"],
                    "skip": True}

        # add masks to the input
        if (add_label):
            question = tokenizer.encode(prompt.format(toxic_text=row["en_toxic_comment"]))
            answer = tokenizer.encode(row["en_neutral_comment"]) + [tokenizer.eos_token_id]
            temp_question = question + answer

            if temp_question != input_ids[-len(temp_question):]:
                import pdb
                pdb.set_trace()
            attention_mask = [1] * len(input_ids)
            loss_mask = [0] * (len(input_ids) - len(answer)) + [1] * len(answer)
        else:
            attention_mask = [1] * len(input_ids)
            loss_mask = [0] * len(input_ids)

        # we prepare the golden label, which must also be tokenized and padded
        golden_label = tokenizer.encode(row["en_neutral_comment"], padding="max_length", max_length=max_seq_len - 1)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "golden_label": golden_label,
                "skip": False}

    prompt = PARADETOX_PROMPT_ONLY_TEMPLATE

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=1,
                                      fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True},
                                      remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=1,
                                    fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False},
                                    remove_columns=test_dataset.column_names)


    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    return train_dataset, test_dataset


def prepare_paradetox_lowdata_ood_dataset(tokenizer, seed, max_seq_len, num_proc):
    # we first read wildguard dataset from huggingface
    train_dataset = read_dataset_to_hf("s-nlp/paradetox")["train"].shuffle(seed=seed)

    def filter_for_test(row):
        list = ["dick", "bitches", "fucks", "nigga", "shits", "retarded", "dumbass", "shitty", "pussies", "lunatic", "crap", "moron", "bastard", "jackass"]
        for word in list:
            if word in row["en_toxic_comment"]:
                return True
        return False

    # create the dataset and filter out specific occurances of words
    test_dataset = train_dataset.filter(filter_for_test)
    train_dataset = train_dataset.filter(lambda row: not filter_for_test(row))

    num_train = 3200
    train_dataset = train_dataset.select(range(num_train))

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        final_instruction = prompt.format(toxic_text=row["en_toxic_comment"])
        if (add_label):
            final_instruction += row["en_neutral_comment"] + tokenizer.eos_token

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "golden_label": row["en_neutral_comment"],
                    "skip": True}

        # add masks to the input
        if (add_label):
            question = tokenizer.encode(prompt.format(toxic_text=row["en_toxic_comment"]))
            answer = tokenizer.encode(row["en_neutral_comment"]) + [tokenizer.eos_token_id]
            temp_question = question + answer

            if temp_question != input_ids[:len(temp_question)]:
                import pdb
                pdb.set_trace()
            attention_mask = [1] * len(input_ids)
            loss_mask = [0] * (len(input_ids) - len(answer)) + [1] * len(answer)
        else:
            attention_mask = [1] * len(input_ids)
            loss_mask = [0] * len(input_ids)

        # we prepare the golden label, which must also be tokenized and padded
        golden_label = tokenizer.encode(row["en_neutral_comment"], padding="max_length", max_length=max_seq_len - 1)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "golden_label": golden_label,
                "skip": False}

    prompt = PARADETOX_PROMPT_ONLY_TEMPLATE

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=1,
                                      fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True},
                                      remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=1,
                                    fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False},
                                    remove_columns=test_dataset.column_names)


    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    return train_dataset, test_dataset

def prepare_paradetox_ood_dataset(tokenizer, seed, max_seq_len, num_proc):
    # we first read wildguard dataset from huggingface
    train_dataset = read_dataset_to_hf("s-nlp/paradetox")["train"].shuffle(seed=seed)

    def filter_for_test(row):
        list = ["dick", "bitches", "fucks", "nigga", "shits", "retarded", "dumbass", "shitty", "pussies", "lunatic", "crap", "moron", "bastard", "jackass"]
        for word in list:
            if word in row["en_toxic_comment"]:
                return True
        return False

    # create the dataset and filter out specific occurances of words
    test_dataset = train_dataset.filter(filter_for_test)
    train_dataset = train_dataset.filter(lambda row: not filter_for_test(row))

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        final_instruction = prompt.format(toxic_text=row["en_toxic_comment"])
        if (add_label):
            final_instruction += row["en_neutral_comment"] + tokenizer.eos_token

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "golden_label": row["en_neutral_comment"],
                    "skip": True}

        # add masks to the input
        if (add_label):
            question = tokenizer.encode(prompt.format(toxic_text=row["en_toxic_comment"]))
            answer = tokenizer.encode(row["en_neutral_comment"]) + [tokenizer.eos_token_id]
            temp_question = question + answer

            if temp_question != input_ids[-len(temp_question):]:
                import pdb
                pdb.set_trace()
            attention_mask = [1] * len(input_ids)
            loss_mask = [0] * (len(input_ids) - len(answer)) + [1] * len(answer)
        else:
            attention_mask = [1] * len(input_ids)
            loss_mask = [0] * len(input_ids)

        # we prepare the golden label, which must also be tokenized and padded
        golden_label = tokenizer.encode(row["en_neutral_comment"], padding="max_length", max_length=max_seq_len - 1)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "golden_label": golden_label,
                "skip": False}

    prompt = PARADETOX_PROMPT_ONLY_TEMPLATE

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=1,
                                      fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True},
                                      remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=1,
                                    fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False},
                                    remove_columns=test_dataset.column_names)

    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    return train_dataset, test_dataset


def prepare_toxigen_prompt_dataset(tokenizer, seed, max_seq_len, num_proc):
    # we first read toxigen dataset from huggingface
    dataset = read_dataset_to_hf("toxigen/toxigen-data", name="prompts")

    # we define in-distribubtion and out-of-distribution datasets
    # for some reason "trans" does not have neutral category, so we omit it from our dataset
    train_categories = ["black", "native_american", "immigrant", "lgbtq", "women", "chinese", "latino", "bisexual", "asian", "mental_disability", "jewish", "muslim"]

    test_categories = ["mexican", "physical_disability", "middle_east"]

    # create the training dataset
    train_dataset_list = []
    for train_category in train_categories:
        category_neutral = dataset[f"neutral_{train_category}_1k"]
        category_hate = dataset[f"hate_{train_category}_1k"]

        found_neutral_sentences = []
        found_hate_sentences = []

        # we loop through examples and select the non-repeating ones
        for i in range(len(category_neutral)):
            text = category_neutral[i]["text"]
            sentences = text.strip("\\n-\n").split("\\n")
            for sentence in sentences:
                sentence = sentence.strip("- ")
                if sentence not in found_neutral_sentences:
                    found_neutral_sentences.append(sentence)

        for i in range(len(category_hate)):
            text = category_hate[i]["text"]
            sentences = text.strip("\\n-\n").split("\\n")
            for sentence in sentences:
                sentence = sentence.strip("- ")
                if sentence not in found_hate_sentences:
                    found_hate_sentences.append(sentence)

        # we now create a new dataset with the non-repeating sentences
        category_neutral = Dataset.from_dict({"text": found_neutral_sentences})
        category_hate = Dataset.from_dict({"text": found_hate_sentences})

        category_neutral = category_neutral.add_column("is_hate", [False] * len(category_neutral))
        category_hate = category_hate.add_column("is_hate", [True] * len(category_hate))
        category_neutral = category_neutral.add_column("category", [train_category] * len(category_neutral))
        category_hate = category_hate.add_column("category", [train_category] * len(category_hate))

        train_dataset_list.append(category_neutral)
        train_dataset_list.append(category_hate)

    test_dataset_list = []
    for test_category in test_categories:
        category_neutral = dataset[f"neutral_{test_category}_1k"]
        category_hate = dataset[f"hate_{test_category}_1k"]

        found_neutral_sentences = []
        found_hate_sentences = []

        # we loop through examples and select the non-repeating ones
        for i in range(len(category_neutral)):
            text = category_neutral[i]["text"]
            sentences = text.strip("\\n-\n").split("\\n")
            for sentence in sentences:
                sentence = sentence.strip("- ")
                if sentence not in found_neutral_sentences:
                    found_neutral_sentences.append(sentence)

        for i in range(len(category_hate)):
            text = category_hate[i]["text"]
            sentences = text.strip("\\n-\n").split("\\n")
            for sentence in sentences:
                sentence = sentence.strip("- ")
                if sentence not in found_hate_sentences:
                    found_hate_sentences.append(sentence)

        # we now create a new dataset with the non-repeating sentences
        category_neutral = Dataset.from_dict({"text": found_neutral_sentences})
        category_hate = Dataset.from_dict({"text": found_hate_sentences})

        category_neutral = category_neutral.add_column("is_hate", [False] * len(category_neutral))
        category_hate = category_hate.add_column("is_hate", [True] * len(category_hate))
        category_neutral = category_neutral.add_column("category", [test_category] * len(category_neutral))
        category_hate = category_hate.add_column("category", [test_category] * len(category_hate))

        test_dataset_list.append(category_neutral)
        test_dataset_list.append(category_hate)

    # we create our training and testing datasets
    train_dataset = concatenate_datasets(train_dataset_list).shuffle(seed=seed)

    test_dataset_id = train_dataset.select(range(int(0.2 * len(train_dataset))))
    train_dataset = train_dataset.select(range(len(test_dataset_id), len(train_dataset)))
    test_dataset_ood = concatenate_datasets(test_dataset_list).shuffle(seed=seed)

    # reformat the dataset
    def reformat_row(row, prompt, labels, tokenizer, add_label=False):
        # we first strip the weird "- " from the beginnings and "\\n-\n" from the end of toxigen prompts
        text = row["text"].strip("- ").strip("\\n-\n")

        final_instruction = prompt.format(input=text, output="")
        if (add_label):
            final_instruction += labels[row["is_hate"]]

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "is_hate": False,
                    "category": "none",
                    "skip": True}

        # add masks to the input
        if (add_label):
            attention_mask = [1] * len(input_ids)
            loss_mask = [0] * (len(input_ids) - 1) + [1]
        else:
            attention_mask = [1] * len(input_ids)
            loss_mask = [0] * len(input_ids)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "is_hate": row["is_hate"],
                "category": row["category"],
                "skip": False}

    prompt = TOXIC_CLASSIFICATION_WITH_PROMPT
    labels = TOXIC_CLASSIFICATION_LABELS

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=1,
                                      fn_kwargs={"prompt": prompt, "labels": labels, "tokenizer": tokenizer, "add_label": True},
                                      remove_columns=train_dataset.column_names)
    test_dataset_id = test_dataset_id.map(reformat_row, batched=False, num_proc=1,
                                    fn_kwargs={"prompt": prompt, "labels": labels, "tokenizer": tokenizer, "add_label": False},
                                    remove_columns=test_dataset_id.column_names)
    test_dataset_ood = test_dataset_ood.map(reformat_row, batched=False, num_proc=1,
                                    fn_kwargs={"prompt": prompt, "labels": labels, "tokenizer": tokenizer, "add_label": False},
                                    remove_columns=test_dataset_ood.column_names)

    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset_id = test_dataset_id.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset_ood = test_dataset_ood.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset_id = test_dataset_id.remove_columns(["skip"])
    test_dataset_ood = test_dataset_ood.remove_columns(["skip"])

    # we now merge test dataset together with id or ood labeling
    test_dataset_id = test_dataset_id.add_column("is_id", [True] * len(test_dataset_id))
    test_dataset_ood = test_dataset_ood.add_column("is_id", [False] * len(test_dataset_ood))

    test_dataset = concatenate_datasets([test_dataset_id, test_dataset_ood])

    return train_dataset, test_dataset


def prepare_tofu_dataset(tokenizer, seed, max_seq_len, num_proc, apply_loss_mask):
    # we first read wildguard dataset from huggingface
    train_dataset = read_dataset_to_hf("locuslab/TOFU", name="retain_perturbed")["train"].shuffle(seed=seed)

    # create tofu_names as a copy of the original tofu names
    tofu_names = TOFU_NAMES.copy()
    tokenized_tofu_names = []

    # add the first and last names of the characters to the tofu names
    for name in TOFU_NAMES:
        tofu_names.append(name.split(" ")[0])
        tofu_names.append(name.split(" ")[-1])

    for name in tofu_names:
        tokenized_tofu_names.append(tokenizer.encode(name))
        tokenized_tofu_names.append(tokenizer.encode(" " + name))

    def filter_for_noname(row):
        # we filter out examples where the dataset doesn't contain the name of the individual
        for name in tofu_names:
            if name in row["question"] or name in row["answer"]:
                return True
        return False

    train_dataset = train_dataset.filter(filter_for_noname)

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, tokenized_tofu_names, apply_loss_mask):
        # final_instruction = prompt.format(question=row["question"], answer=row["answer"])
        final_instruction = row["answer"] # we only train on the answer

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        # input_ids = tokenizer.encode(final_instruction)

        stringified_input_ids = " ".join([str(x) for x in input_ids])

        # we select out all the sensitive name tokens
        for name_tokens in tokenized_tofu_names:
            stringified_name_tokens = " ".join([str(x) for x in name_tokens])
            while stringified_name_tokens in stringified_input_ids:
                # remove the name tokens from the input_ids
                occurence = stringified_input_ids.index(stringified_name_tokens)
                stringified_input_ids = stringified_input_ids[:occurence] + stringified_input_ids[occurence + len(stringified_name_tokens) + 1:]

        filtered_input_ids = [int(x) for x in stringified_input_ids.split(" ")]

        loss_mask = []
        # we now create the loss mask by checking if the token was filtered out or not
        orig_input_ids_ptr = 0
        filtered_input_ids_ptr = 0
        while (orig_input_ids_ptr < len(input_ids)):
            if (input_ids[orig_input_ids_ptr] == filtered_input_ids[filtered_input_ids_ptr]):
                loss_mask += [1]
                orig_input_ids_ptr += 1
                filtered_input_ids_ptr += 1
            else:
                loss_mask += [0]
                orig_input_ids_ptr += 1

        # set attention mask to exclude pad token
        attention_mask = [1 if i != tokenizer.pad_token_id else 0 for i in input_ids]

        # [1] * len(input_ids))
        if not apply_loss_mask:
            loss_mask = [1] * len(input_ids)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "skip": True}

        # we prepare the golden label, which must also be tokenized and padded
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "skip": False}

    prompt = TOFU_TEMPLATE

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=1,
                                      fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "tokenized_tofu_names": tokenized_tofu_names, "apply_loss_mask": apply_loss_mask},
                                      remove_columns=train_dataset.column_names)

    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])

    return train_dataset


def prepare_dataset_for_training(data_to_load, tokenizer, seed, num_proc, max_seq_len, apply_loss_mask):
    """Load and reformat a dataset for training
    """
    if apply_loss_mask == False:
        if data_to_load != "tofu":
            raise ValueError("no loss mask only implemented for tofu")

    if data_to_load == "wildguard_prompt":
        # we read the wildguard dataset from huggingface
        train_dataset, test_dataset = prepare_wildguard_prompt_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset = {data_to_load: test_dataset} # to turn it into an eval dictionary for hf trainer
        return train_dataset, eval_dataset
    if data_to_load == "paradetox":
        # we read paradetox dataset from huggingface
        train_dataset, test_dataset = prepare_paradetox_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset = {data_to_load: test_dataset} # to turn it into an eval dictionary for hf trainer
        return train_dataset, eval_dataset
    if data_to_load == "paradetox_ood":
        # we read paradetox dataset from huggingface
        train_dataset, test_dataset = prepare_paradetox_ood_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset = {data_to_load: test_dataset} # to turn it into an eval dictionary for hf trainer
        return train_dataset, eval_dataset
    if data_to_load == "paradetox_lowdata":
        # we read paradetox dataset from huggingface
        train_dataset, test_dataset = prepare_paradetox_lowdata_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset = {data_to_load: test_dataset}
        return train_dataset, eval_dataset
    if data_to_load == "paradetox_lowdata_ood":
        train_dataset, test_dataset = prepare_paradetox_lowdata_ood_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset = {data_to_load: test_dataset}
        return train_dataset, eval_dataset
    if data_to_load == "toxigen_finetune_prompts":
        train_dataset, test_dataset = prepare_toxigen_prompt_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset = {data_to_load: test_dataset}
        return train_dataset, eval_dataset
    if data_to_load == "tofu":
        train_dataset = prepare_tofu_dataset(tokenizer, seed, max_seq_len, num_proc, apply_loss_mask)
        # note: we do not do in-training evaluation for tofu
        return train_dataset, {}
    else:
        raise ValueError(f"Unknown dataset: {data_to_load}")
