## Install environment

Make sure you have Python>=3.9 and Cuda>=11.2 installed. Then run:
```bash
pip install -r requirements.txt
```

## Preprocess data
Adapt the config file in `configs/preprocess_owt2.json` to your specific case, and then run:
```
python preprocess.py --config=configs/your_config_file.json
```

## Training
### Encoder
To train an encoder model:
1. Write/edit model-related parameters in a config file similar to `configs/mlm_headless.json`
2. Run the following command with your specific arguments:
```bash
python mlm_headless.py \
    --config configs/your_config_file.json \
    --num_nodes your-gpu-node-count \
    --global_bs your-accumulated-batch_size \
    --gpu_bs your-per-device-batch-size \
    --dataset your-preprocessed-output.hf \
    --hf_tokenizer your-tokenizer \
    --hf_path path-to-your-model-arch-on-HF \
    --model_max_seq_len models-max-pos-embeddings \
    --run_name run-name-for-logging-and-ckpts \
    --saved_ckpt_path where-to-save-ckpts
```
Other args include `--accelerator` (`hf`, `xformers` or `flash_attention`), `--ckpt_every` to pick checkpoint frequency, among others.

3. Pick your checkpoint and publish it to HuggingFace:
```python
python hf_publisher.py \
    --hf_name your_hf_id/your_model \
    --model_ckpt your_model.ckpt \
    --mode mlm
```
### Decoder
To train a decoder model:
1. Write/edit model-related parameters in a config file similar to `configs/gpt_headless_70m.json`
2. Run the following command with your specific arguments:
```bash
python gpt_headless.py \
    --config configs/your_config_file.json \
    --num_nodes your-gpu-node-count \
    --global_bs your-accumulated-batch_size \
    --gpu_bs your-per-device-batch-size \
    --dataset your-preprocessed-output.hf \
    --hf_tokenizer your-tokenizer \
    --hf_path path-to-your-model-arch-on-HF \
    --model_max_seq_len models-max-pos-embeddings \
    --run_name run-name-for-logging-and-ckpts \
    --saved_ckpt_path where-to-save-ckpts
```
Other args include `--accelerator` (`hf`, `xformers` or `flash_attention`), `--ckpt_every` to pick checkpoint frequency, among others.

3. (optional) Pick your checkpoint and publish it to HuggingFace. You'll need to use the `add_head` option to make it able to output tokens:
```python
python hf_publisher.py \
    --hf_name your_hf_id/your_model \
    --model_ckpt your_model.ckpt \
    --mode add_head
```

4. The resulting model will probably perform poorly for language generation. Why? Because it was not trained to do it! To turn your contrastive model into a good LM, you'll need add a head and fine-tune it. Setup a config file in the style of `config/gpt_vanilla_ft.json` and run:
```
python ft_gpt_headless.py \
    --ckpt_path your_headless_model.ckpt' \
    --config configs/your_ft_config.json \
    --num_nodes your-gpu-nodes \
    --global_bs your-accumulated-bs \
    --gpu_bs your-device-bs \
    --dataset your-preprocessed-output.hf \
    --run_name run-name-for-logging-and-ckpts \
    --saved_ckpt_path where-to-save-finetuned-ckpts
```

5. Pick your fine-tuned checkpoint and publish it to HuggingFace. You don't need to use the `add_head` option anymore as you just trained one:
```python
python hf_publisher.py \
    --hf_name your_hf_id/your_model \
    --model_ckpt your_model.ckpt \
    --mode lm
```

## Evaluation
You can now use any zero-shot or fine-tuning code to evaluate your models. We provide our GLUE fine-tuning script in `glue_finetuning.py`, and we used the [LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness) for zero-shot evaluation.