from dotenv import load_dotenv
import os

import openai
from langchain_community.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import transformers


# Load environment variables from .env file
load_dotenv()


def load_llm_pipeline(model_id: str):
    """Used to load a pipeline object from huggingface transformers library for a given model_id
    to cache for future local use.

    Args:
        model_id (str): model id from huggingface model hub

    Returns:
        pipeline object from huggingface transformers library
    """
    huggingface_token = os.getenv("HUGGINGFACE_API_KEY")
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token=huggingface_token
    )
    
    return pipeline


# Llama family interface
def llama(model, prompt: str, temperature=0.7) -> str:
    """Uses the Llama family model to generate text based on a prompt that is preloaded in the model argument.

    Args:
        model : a preloaded Llama model
        prompt (str): the prompt to generate text from
        temperature (float, optional): model temperature. Defaults to 0.7.

    Returns:
        str: generated text
    """
    # llama requires non-zero temperature
    if temperature == 0.0:
        temperature = 0.0001
        
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": prompt},
    ]

    final_prompt = model.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )

    terminators = [
        model.tokenizer.eos_token_id,
        model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model(
                prompt,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                )
    
    return outputs[0]["generated_text"][len(prompt):]


# GPT api interface
def gpt(model: str, prompt: str, temperature=0.7) -> str:
    """ Make API call to OpenAI API to generate text using GPT model for a given prompt

    Args:
        model (str): OpenAI model ID
        prompt (str): User prompt
        temperature (float, optional): model temperature. Defaults to 0.7.

    Returns:
        str: Generated text
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # api call to OpenAI
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    
    return response.choices[0].message.content # return generated text