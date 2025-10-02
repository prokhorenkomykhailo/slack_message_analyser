#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
from typing import List, Dict

def load_model(model_id: str):
    """
    Load the 4-bit quantized model using BitsAndBytesConfig.
    Returns (tokenizer, model).
    """
    # Create a BitsAndBytesConfig that enables 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",  # or another quant type depending on model
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Loading model in 4-bit mode …")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        # optionally you can add `offload_folder` or `offload_state_dict` if memory is tight
    )
    print("Model loaded.")
    return tokenizer, model

def prepare_prompt(messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
    """
    Given a list of messages (chat format), prepare input_ids for generation.
    Each message is a dict: {"role": "...", "content": "..."}.
    """
    # Use the chat template logic embedded in the tokenizer
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    return inputs

def generate_response(
    input_ids: torch.Tensor,
    model,
    tokenizer,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    top_p: float = 0.9,
    do_sample: bool = False
) -> str:
    with torch.no_grad():
        gen = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Only take the newly generated portion
    # gen is shape (1, seq_len_total). We want tokens after the input length.
    input_len = input_ids.shape[-1]
    new_tokens = gen[0, input_len:]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return output

if __name__ == "__main__":
    # Change this to the proper model ID (be sure you have access)
    model_id = "CohereLabs/c4ai-command-r-plus-4bit"  # or "CohereForAI/..." if that’s the correct namespace

    # Load model + tokenizer
    tokenizer, model = load_model(model_id)

    # Example chat messages
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]

    # Prepare the prompt
    inputs = prepare_prompt(messages)

    # Generate a response
    reply = generate_response(
        input_ids=inputs["input_ids"].to(model.device),
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.3,
        top_p=0.9,
        do_sample=True
    )

    print("=== Reply ===")
    print(reply)

    # (Optional) wrap into JSON or structured output
    out = {
        "model": model_id,
        "messages": messages,
        "response": reply
    }
    print(json.dumps(out, indent=2))
