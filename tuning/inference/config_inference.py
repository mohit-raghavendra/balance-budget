from pydantic import BaseModel

class VLLMSamplingParamsConfig(BaseModel):
    max_tokens: int = 4096
    temperature: float = 0.5
    top_k: int = 150
    top_p: float = 0.9
    stop: list[str] = ["<|im_end|>", "<|end_of_text|>"]
    # repetition_penalty: float = 1.1


if __name__ == "__main__":
    print({**VLLMSamplingParamsConfig().model_dump()})
