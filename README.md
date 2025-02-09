# decouple project

## Environment Setup
```
conda create -n decouple_shared python=3.10
conda activate decouple_shared
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Masked training

### Tofu experiment
Make sure you are on the root directory. To run the tofu experiment, run the following in the conda environment

```angular2html
bash scripts/train_hf.sh
```

This will train a model and store it in a new directory called `models`. 

### Understanding the code

The core logic for implementing the loss masking is in `src/SelectiveLossTrainer.py`. Intuitively, 
- in addition to "input_ids" and "attention_mask", I add a new boolean input field called "loss_mask" that has the same dimensions as the input_ids
- a token whose corresponding loss_mask is 0 is masked out, while tokens with loss_mask 1 are normally backpropagated for training

### Adding new tasks or changing current tasks

New experiments can be implemented by editing the `prepare_dataset_for_training` method that is called in the `src/run_train_hf.py` script. This method is core to deciding which dataset to load and which tokens to mask out. The rest of the training script is general purpose and will train normally. 


## Evaluation
There are two ways to evaluate the model. 

The first way is to evaluate the model while it's training. To implement this, you should modify the "prepare_dataset_for_training" to prepare an evaluation dataset to pass to the Huggingface Trainer. You will also need to edit the `evaluate` method in the `src/SelectiveLossTrainer.py` to implement the evaluation pipeline. This method of evaluation is not used by the Tofu experiment.  

The second way is to evaluate the model after training is completed, using a seperate script. The tofu experiment uses this second evaluation method. Evaluate on a trained model after training is completed by running the following command in the conda environment, which will run the tofu evaluations. 

```angular2html
bash scripts/eval_hf.sh
```

