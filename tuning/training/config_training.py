from enum import Enum
from pydantic import BaseModel
from tuning.config import MODELS_DIR
from typing import Optional

BaseModel.model_config['protected_namespaces'] = ()


def sft_batch_size(dataset_size: int):
    return 1

def dpo_batch_size(dataset_size: int):
    return 1

def effective_batch_size(dataset_size: int):
    return 16

class ModelLoadConfig(BaseModel):
    max_seq_length: str = 1024 
    dtype: str = None 
    load_in_4bit: bool = False 

class LoraConfig(BaseModel):
    r: int = 32
    target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",]
    lora_alpha: int = 32
    lora_dropout: int = 0
    bias: str = "none"
    use_gradient_checkpointing: bool = False
    random_state: int = 42
    use_rslora: bool = False
    loftq_config: str = None

class TrainingArgumentsConfig(BaseModel):
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    eval_steps: float = 0.1
    logging_steps: int = 1
    do_eval: bool = True
    eval_strategy: str = "steps"
    gradient_accumulation_steps: int = 4
    warmup_ratio: int = 0.1
    num_train_epochs: int = 2
    learning_rate: float = 5e-5
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    report_to: list[str] = ["wandb"]


class DPOTrainingConfig(TrainingArgumentsConfig):
    beta: float = 1
    learning_rate: float = 5e-6
    num_train_epochs: int = 2
    per_device_eval_batch_size: int = 2


class DatasetConfig(BaseModel):
    dataset: str = "gsm8k"
    dataset_type: str = "sft"
    train_size: int = 100

    @property
    def dataset_full_name(self):
        if not self.train_size:
            return f"{self.dataset_type}-{self.dataset}"
        return f"{self.dataset_type}-{self.dataset}-{self.train_size}"
    
    def __str__(self):
        return self.dataset_full_name

    
class SFTRunConfig(BaseModel):
    model_name_hf: str = "unsloth/Meta-Llama-3.1-8B"
    dataset_config: Optional[DatasetConfig] = None
    model_name: str = "llama3-8B"
    task_name: str = "math"
    run_type: str = "sft"
    do_training: bool = False
    do_inference: bool = False
    do_evaluation: bool = False
    
    @property
    def run_name(self):
        if not self.dataset_config or not self.dataset_config.train_size:
            return self.model_name
        return f"{self.model_name}_{self.dataset_config.dataset_full_name}"
    
    @property
    def output_dir(self):
        return f"{MODELS_DIR}/{self.run_name}"
    
    def __str__(self):
        return self.run_name
    

class PTRunConfig(BaseModel):
    model_name_hf: str = "unsloth/Meta-Llama-3.1-8B"
    model_name: str = "llama3-8B"
    dataset_config: DatasetConfig = None
    sft_run_config: Optional[SFTRunConfig] = None
    run_type: str = "pt"
    task_name: str = "math"
    do_training: bool = False
    do_inference: bool = False
    do_evaluation: bool = False
    pft_method: str = "dpo"
    add_beta_run_name: bool = False
    beta: float = 0.1

    @property
    def run_name(self):
        run_name = self.model_name
        if self.sft_run_config:
            run_name = self.sft_run_config.run_name
        if self.dataset_config:
            run_name = f"{run_name}_{self.dataset_config.dataset_full_name}"
        if self.pft_method == "kto":
            run_name = f"{run_name}_{self.pft_method}"
        if self.add_beta_run_name:
            run_name = f"{run_name}_beta-{self.beta}"

            run_name = run_name.replace(".", "-")   
        return run_name
    
    @property
    def output_dir(self):
        return f"{MODELS_DIR}/{self.run_name}"
    
    def __str__(self):
        return self.run_name