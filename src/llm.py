# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from peft import PeftModel
# from langchain_huggingface import HuggingFacePipeline

# BASE_MODEL = "unsloth/llama-2-7b-chat-bnb-4bit"
# ADAPTER_PATH = "llama-medical-llm"

# def load_finetuned_llm():
#     print("Loading tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

#     print("Loading base model in 4-bit...")
#     # bnb_config = BitsAndBytesConfig(
#     #     load_in_4bit=True,
#     #     bnb_4bit_quant_type="nf4",
#     #     bnb_4bit_compute_dtype=torch.float16,
#     # )

#     base_model = AutoModelForCausalLM.from_pretrained(
#         BASE_MODEL,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         low_cpu_mem_usage=True,
#     )

#     print("Merging LoRA adapter...")
#     model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
#     model.eval()

#     print("Creating pipeline...")
#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=512,
#         temperature=0.2,
#         do_sample=True,
#         repetition_penalty=1.1,
#     )

#     llm = HuggingFacePipeline(pipeline=pipe)
#     print("LLM ready.")
#     return llm

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def load_llm():
    llm = ChatGroq(model='llama-3.3-70b-versatile', max_retries=10)

    return llm