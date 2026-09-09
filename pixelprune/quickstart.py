import os
os.environ["PIXELPRUNE_ENABLED"] = "true"

from pixelprune import apply_pixelprune
apply_pixelprune(model="qwen3_5_moe")  # Qwen3.5 MoE
# apply_pixelprune(model="qwen3_vl")  # or "qwen3_5" for Qwen3.5  
# call BEFORE loading the model

from transformers import AutoModelForImageTextToText, AutoProcessor

model_name = "/mnt/nfs/data/pretrained_models/Qwen3.5-35B-A3B/"  # or Qwen3.5 model path
model = AutoModelForImageTextToText.from_pretrained(
    model_name, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

# Document parsing example
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "assets/doc.jpg"},
            {"type": "text", "text": "Parse this table into Markdown format."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
print(processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0])
