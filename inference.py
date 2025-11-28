from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import torch

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./llama-ncert-3b/final"

# Detect Apple Silicon backend
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(base_model)

# Use float32 for MPS (16-bit not stable)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    device_map=None
).to(device)

# Apply LoRA (if available)
try:
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    print("✅ LoRA adapter loaded.")
except Exception as e:
    print(f"⚠️ LoRA adapter failed: {e}")

# --- Stopping criteria ---
class MultiStopCriteria(StoppingCriteria):
    def __init__(self, stop_sequences_ids):
        super().__init__()
        self.stop_sequences_ids = stop_sequences_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0]
        for stop_ids in self.stop_sequences_ids:
            L = len(stop_ids)
            if L and seq.size(0) >= L and seq[-L:].tolist() == stop_ids:
                return True
        return False


def generate_mcq(prompt, max_new_tokens=80, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    stop_strings = ["Answer:", "Ans:", "\nQ:", "\n\nQ:", "\n1.", "\nQ1:"]
    stop_ids_list = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]
    stopping_criteria = StoppingCriteriaList([MultiStopCriteria(stop_ids_list)])

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        stopping_criteria=stopping_criteria,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )

    gen_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    for s in stop_strings:
        if s in gen_text:
            gen_text = gen_text.split(s)[0].strip()
    return gen_text


if __name__ == "__main__":
    prompt = "Generate a multiple-choice question (MCQ) in science at difficulty 3."
    print(generate_mcq(prompt))