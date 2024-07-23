import transformers
import torch

model_id = "42dot/42dot_LLM-SFT-1.3B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16}
)

pipeline.model.eval()

prompt = "삼성전자에 대해 알려줘."

outputs = pipeline(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    top_p=0.9
)

print(outputs[0]['generated_text'][len(prompt):])