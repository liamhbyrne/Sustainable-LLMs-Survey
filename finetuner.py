import torch
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
    prepare_model_for_int8_training,
)


class PEFTFineTuner:
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
            max_length=256,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def prepare_datasets(self):
        dataset = load_dataset(self._dataset_name, split="train")
        # Apply the function to all elements in the dataset
        dataset = dataset.map(PEFTFineTuner.merge_columns, remove_columns=['instruction', 'input', 'output'])
        dataset = dataset.select(range(5000))

        train = dataset.train_test_split(test_size=0.1)["train"]
        test = dataset.train_test_split(test_size=0.1)["test"]

        self._train_dataset = train.map(self.tokenize)
        self._eval_dataset = test.map(self.tokenize)
        # Print out summary of the dataset
        print(self._train_dataset)
        print(self._eval_dataset)

    def prepare_4bit_model(self):
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

    def prepare_training_args(self):
        batch_size = 2
        per_device_train_batch_size = 2
        gradient_accumulation_steps = batch_size // per_device_train_batch_size
        output_dir = "irp"
        self._training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=1,
            warmup_steps=100,
            learning_rate=3e-4,
            fp16=True,
            logging_steps=10,
            optim="paged_adamw_32bit",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=20,
            save_steps=20,
            output_dir=output_dir,
            group_by_length=True,
            report_to="wandb",
            run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        )

    def train(self):
        self._model.train()
        self._model = prepare_model_for_int8_training(self._model)


        config = LoraConfig(
            r=16,
            lora_alpha=16,
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

        if torch.cuda.device_count() > 1:
            self.model.is_parallelizable = True
            self.model.model_parallel = True

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
        old_state_dict = self._model.state_dict
        self._model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
            self._model, type(self._model)
        )
        if torch.__version__ >= "2" and sys.platform != "win32":
            print("compiling the model")
            self._model = torch.compile(self._model)

        # GO!
        self._trainer.train()


if __name__ == "__main__":
    fine_tuner = PEFTFineTuner(base_model_name="codellama/CodeLlama-7b-hf", dataset_name="sahil2801/CodeAlpaca-20k", wandb_project="IRP")

    fine_tuner.clear_cache()
    fine_tuner.prepare_4bit_model()
    fine_tuner.prepare_datasets()
    fine_tuner.prepare_training_args()
    fine_tuner.train()
