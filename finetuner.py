from datetime import datetime
import os
import sys
import gc
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    prepare_model_for_int8_training 
)
import torch


class PEFTFineTuner:
    """
    Fine-tunes a model using the PEFT method.
    """
    def __init__(self, base_model_name: str = None, dataset_name: str = None, wandb_project: str = None):
        self._base_model_name = base_model_name
        self._dataset_name = dataset_name
        self._tokenizer = None
        self._model = None
        self._trainer = None
        self._wandb_project = wandb_project
        os.environ["WANDB_PROJECT"] = self._wandb_project
        self._train_dataset = None
        self._eval_dataset = None
        self._training_args = None

    def clear_cache(self):
        """
        Clears the cache and empties the GPU memory.
        """
        self._model = None
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def merge_columns(example):
        if example['input']:
            merged = f"<s>[INST] <<SYS>>\nBelow is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n<</SYS>>\n\n{example['instruction']} Input: {example['input']} [/INST] {example['output']} </s>"
        else:
            merged = f"<s>[INST] <<SYS>>\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.\n<</SYS>>\n\n{example['instruction']} [/INST] {example['output']} </s>"
        return {"text": merged}

    def tokenize(self, sample):
        prompt = sample['text']
        result = self._tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def prepare_datasets(self):
        """
        Loads the dataset and tokenizes it. 
        Assumes that the dataset is in JSON format and hosted locally due to Iridis-5 compute nodes lack internet access.
        """
        print("Loading dataset...")

        dataset = load_dataset("json", data_files=self._dataset_name, split="train")
        # Apply the function to all elements in the dataset
        dataset = dataset.map(PEFTFineTuner.merge_columns, remove_columns=['instruction', 'input', 'output'])

        train = dataset.train_test_split(test_size=0.1)["train"]
        test = dataset.train_test_split(test_size=0.1)["test"]

        self._train_dataset = train.map(self.tokenize)
        self._eval_dataset = test.map(self.tokenize)
        # Print out summary of the dataset
        print(self._train_dataset)
        print(self._eval_dataset)
        print("Dataset loaded.")

    def prepare_4bit_model(self):
        """
        Loads the model with 4-bit quantization for QLoRA.
        """
        print("Loading model with 4-bit quantization...")

        self._tokenizer = AutoTokenizer.from_pretrained(self._base_model_name)
        self._tokenizer.add_eos_token = True
        self._tokenizer.pad_token_id = 0
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            self._base_model_name,
            device_map='auto',
            torch_dtype=torch.bfloat16, 
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
        )
        self._model.train()
        self._model = prepare_model_for_kbit_training(self._model)
        print("Model loaded.")

    def prepare_int8_model(self):
        """
        Loads the model with 8-bit quantization for 8-bit LoRA.
        """
        print("Loading model with 8-bit quantization...")

        self._tokenizer = AutoTokenizer.from_pretrained(self._base_model_name)
        self._tokenizer.add_eos_token = True
        self._tokenizer.pad_token_id = 0
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            self._base_model_name,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            torch_dtype=torch.float16,  # Computations done in float16
            device_map="auto",
        )
        self._model.train()
        self._model = prepare_model_for_int8_training(self._model)
        print("8 bit Model loaded.")
    
    def prepare_16bit_model(self):
        """
        Loads the model with 16-bit quantization for LoRA.
        """
        print("Loading model with 16-bit ...")

        self._tokenizer = AutoTokenizer.from_pretrained(self._base_model_name)
        self._tokenizer.add_eos_token = True
        self._tokenizer.pad_token_id = 0
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            self._base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self._model.train()
        print("Model loaded.")

    def prepare_training_args(self, batch_size: int = 8, run_name: str = None):
        """
        Set up the training arguments for the model.
        """
        print("Setting up training arguments...")
        output_dir = f"/scratch/lhb1g20/checkpoints/{run_name}{datetime.now().strftime('%Y-%m-%d-%Hh%Mm')}"
        self._training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            num_train_epochs=3,
            warmup_steps=100,
            learning_rate=3e-4,
            # fp16=True,
            logging_steps=10,
            optim="paged_adamw_32bit",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=100,
            output_dir=output_dir,
            group_by_length=True,
            report_to="wandb",
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        )
        print("Training arguments set.")
    
    @staticmethod
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def train(self):
        """
        Trains the model.
        """
        print("Training model...")

        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(self._model, config)

        PEFTFineTuner.print_trainable_parameters(self._model)

        if torch.cuda.device_count() > 1:
            self._model.is_parallelizable = True
            self._model.model_parallel = True

        self._trainer = Trainer(
            model=self._model,
            train_dataset=self._train_dataset,
            eval_dataset=self._eval_dataset,
            args=self._training_args,
            data_collator=DataCollatorForSeq2Seq(
                self._tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )

        # Pytorch Optimizations
        self._model.config.use_cache = False

        # GO!
        self._trainer.train()


if __name__ == "__main__":
    root_model_path = "/scratch/lhb1g20/" 

    SIZE = "13b"

    if SIZE == "13b":
        base_model = f"{root_model_path}CodeLlama-13b-hf"
    elif SIZE == "7b":
        base_model = f"{root_model_path}CodeLlama-7b-hf"
    
    fine_tuner = PEFTFineTuner(
        base_model_name=base_model,
        dataset_name="/scratch/lhb1g20/CodeAlpaca-20k/code_alpaca_20k.json",
        wandb_project="IRP"
    )

    fine_tuner.clear_cache()

    BITS = 8
    if BITS == 4:
        fine_tuner.prepare_4bit_model()
    elif BITS == 8:
        fine_tuner.prepare_int8_model()
    elif BITS == 16:
        fine_tuner.prepare_16bit_model()

    fine_tuner.prepare_datasets()

    BATCH_SIZE = 2
    RUN_NAME = "EXPERIMENT-2-LoRA-int8-13B-bat2"
    print(f"CONFIGURATION for {RUN_NAME}:\nSize {SIZE}\nBatch {BATCH_SIZE}\n Bits {BITS}")
    fine_tuner.prepare_training_args(batch_size=BATCH_SIZE, run_name=RUN_NAME)
    fine_tuner.train() 
