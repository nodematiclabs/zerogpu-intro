import gradio as gr
import huggingface_hub
import os
import spaces
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

@spaces.GPU
def sentience_check():
    huggingface_hub.login(token=os.environ["HUGGINGFACE_TOKEN"])
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it").to(device)

    inputs = tokenizer("Are you sentient?", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=128, pad_token_id = tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

demo = gr.Interface(fn=sentience_check, inputs=None, outputs=gr.Text())
demo.launch()
