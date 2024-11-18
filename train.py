# Code adapted from https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py
# and https://huggingface.co/blog/gemma-peft
import multiprocessing
import os
import hydra

import torch
import transformers
from accelerate import PartialState
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    is_torch_npu_available,
    is_torch_xpu_available,
    logging,
    set_seed,
)
from trl import SFTTrainer

def train(cfg):
    # config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    bnb_config = None
    if cfg.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map={"": PartialState().process_index},
        attention_dropout=cfg.attention_dropout,
    )
    
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # load dataset
    data = Dataset.load_from_disk(cfg.dataset_path)

    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        max_seq_length=cfg.max_seq_length,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=cfg.micro_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            gradient_checkpointing=cfg.gradient_checkpointing,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            weight_decay=cfg.weight_decay,
            bf16=cfg.bf16,
            logging_strategy="steps",
            logging_steps=10,
            output_dir=cfg.output_dir,
            optim="paged_adamw_8bit",
            seed=cfg.seed,
            run_name=f"train-{cfg.model_id.split('/')[-1]}",
            report_to="wandb",
        ),
        peft_config=lora_config,
        dataset_text_field=cfg.dataset_text_field,
    )

    # launch
    print("Training...")
    trainer.train()

    print("Saving the last checkpoint of the model")
    trainer.save_model(os.path.join(cfg.output_dir, "final_checkpoint/"))
    # model.save_pretrained(os.path.join(cfg.output_dir, "final_checkpoint/"))

    if cfg.save_merged_model:
        # Free memory for merging weights
        del model
        if is_torch_xpu_available():
            torch.xpu.empty_cache()
        elif is_torch_npu_available():
            torch.npu.empty_cache()
        else:
            torch.cuda.empty_cache()

        output_final_dir = os.path.join(cfg.output_dir, "final_checkpoint/")
        model = AutoPeftModelForCausalLM.from_pretrained(output_final_dir, device_map="auto", torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()

        output_merged_dir = os.path.join(cfg.output_dir, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)

        if cfg.push_to_hub:
            model.push_to_hub(cfg.repo_id, "Upload model")
    
    print("Training Done! 💥")

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    logging.set_verbosity_error()
    train(cfg)

if __name__ == "__main__":
    main()
