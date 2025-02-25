from tuning.utils.utils import chat_template_func
from vllm import LLM, SamplingParams
from tuning.config import MODELS_DIR
from tuning.inference.config_inference import VLLMSamplingParamsConfig


def load_vlm_model(model_name: str) -> LLM:
    model_path = f"{MODELS_DIR}/{model_name}"
    print(f"Loading model from {model_path}")
    
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(
        **VLLMSamplingParamsConfig().model_dump()
    )

    return llm, sampling_params

def make_vllm_call(llm: LLM, sampling_params: SamplingParams, prompts: list[str]) -> list[str]:
        
    tokenizer = llm.get_tokenizer()
    tokenizer = chat_template_func(tokenizer)
    chat_template = tokenizer.chat_template

    outputs = llm.chat(prompts, sampling_params, chat_template=chat_template)
    if sampling_params.n == 1:
        responses = [output.outputs[0].text for output in outputs]
    else:
        responses = [[response.text for response in output.outputs] for output in outputs]

    print(f"Generated {len(responses)} responses using vllm")

    return responses

def tokenize_test_dataset(llm, messages):

    tokenizer = llm.get_tokenizer()
    tokenizer = chat_template_func(tokenizer)
    tokenized_prompts = [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]

    return tokenized_prompts

def generate_responses_vllm(llm: LLM, sampling_params: SamplingParams, prompts: list[str], dataset):

    responses = make_vllm_call(llm, sampling_params, dataset)

    results = [
        {
            "prompt": prompt,
            "response": response,
        }
        for prompt, response in zip(prompts, responses)
    ]
    return results
