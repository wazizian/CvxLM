from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def make_inference(msg: str, ckpt: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to(device)
    inputs = tokenizer.encode(msg, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=300, temperature=0.2, top_p=0.9, do_sample=True, num_beams=5)
    return tokenizer.decode(outputs[0])

def main():
    msg = """
here is the python cvxpy code that solves the linear programming problem:
    """
    models  = [
        "HuggingFaceTB/SmolLM2-360M",
        "finetune_smollm2_360M_cvxpy/final_checkpoint",
        "HuggingFaceTB/SmolLM2-1.7B",
        "finetune_smollm2_1.7B_cvxpy/final_checkpoint",
        "Qwen/Qwen2.5-Coder-0.5B",
        "finetune_qwen2.5_0.5B_cvxpy/final_checkpoint",
        ]
    print(f"Messsage: {msg}")
    print("-" * 50)
    for model in models:
        print(f"Model: {model}")
        print(make_inference(msg, model))
        print("-" * 50)

if __name__ == "__main__":
    main()
