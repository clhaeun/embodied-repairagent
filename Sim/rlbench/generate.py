import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_code_with_local_llm(prompt: str, model_name="Qwen/Qwen2.5-Coder-7B-Instruct"):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    messages = [
        {"role": "system", "content": "You are an expert robotics code generator."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=2048,
            temperature=0.2,
            do_sample=True,
            top_p=0.95,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # assistant 이후만 추출
    if "assistant" in output_text:
        output_text = output_text.split("assistant", 1)[-1].strip()

    return output_text