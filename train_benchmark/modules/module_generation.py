from modules.utils import *
from trl import SFTTrainer
from datasets import Dataset
from accelerate import Accelerator
from tqdm import tqdm

class LLMTextGeneration:
    def __init__(self, model_name, tokenizer_max_length=128,  max_new_tokens=4, dtype_str='bfloat16', device='auto'):
        # model_name = add_path_prefix(model_name)
        self.model_name = model_name
        self.tokenizer_max_length = tokenizer_max_length
        self.max_new_tokens = max_new_tokens
        self.saved_model_path = None
        # Initialize the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.device = device
        self.dtype = get_torch_dtype(dtype_str)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype=self.dtype, device_map=self.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.accelerator = Accelerator()
        self.model = self.accelerator.prepare(self.model)
        # self.model.to(self.device) #Accelerator aleady handles device placement.
        
        

    def formatting_func_inference(self, question):
        system_prompt = "You are a helpful assistant."
        if self.model_name.endswith("meta-llama/Llama-2-7b-chat-hf") or self.model_name.endswith("princeton-nlp/Sheared-LLaMA-1.3B"):
            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{question} [/INST]"
        elif self.model_name.endswith("meta-llama/Meta-Llama-3-8B"):
            prompt = f"<|begin_of_text|>{question}"
        elif self.model_name.endswith("meta-llama/Meta-Llama-3-8B-Instruct") or self.model_name.endswith("meta-llama/Meta-Llama-3-70B-Instruct"):
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        else:
            # prompt = ""
            # raise Exception("Model not recognized!!")
            prompt = question
        return prompt

    def inference(self, questions):
        '''
        Llama2 format: f"<s>[INST] <<SYS>>\n{prompt_system_message}\n<</SYS>>\n\n{question} [/INST]"
        Llama3 base template: <|begin_of_text|>{{ user_message }}
        Llama3 chat template:
                            <|begin_of_text|>
                            <|start_header_id|>system<|end_header_id|>{{ system_prompt }}<|eot_id|>
                            <|start_header_id|>user<|end_header_id|>{{ user_message }}<|eot_id|>
                            <|start_header_id|>assistant<|end_header_id|>
        '''
        pipeline = transformers.pipeline(
                "text-generation",
                model=self.model,
                torch_dtype=self.dtype,
                tokenizer=self.tokenizer,  # It's also good practice to specify the tokenizer explicitly
                # device=self.device,  # Specify the device to use; 0 usually refers to the first GPU
                # device_map="auto",
            )

        terminators = [pipeline.tokenizer.eos_token_id]
        if "Llama-3" in self.model_name: terminators.append(pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        responses = []
        prompts = [self.formatting_func_inference(question) for question in questions]
        results = pipeline(
            prompts,
            do_sample=False, 
            num_return_sequences=1,
            return_full_text=False,
            eos_token_id=terminators,
            # max_length=self.tokenizer_max_length,
            # truncation=True,
            max_new_tokens=self.tokenizer_max_length,

        )
        for result in results:
            responses.append(result[0]['generated_text'].lstrip())
        return responses



    def inference_single(self, questions): #Use pipeline for 1 example at a time
        '''
        Llama2 format: f"<s>[INST] <<SYS>>\n{prompt_system_message}\n<</SYS>>\n\n{question} [/INST]"
        Llama3 base template: <|begin_of_text|>{{ user_message }}
        Llama3 chat template:
                            <|begin_of_text|>
                            <|start_header_id|>system<|end_header_id|>{{ system_prompt }}<|eot_id|>
                            <|start_header_id|>user<|end_header_id|>{{ user_message }}<|eot_id|>
                            <|start_header_id|>assistant<|end_header_id|>
        XM note: if large amount of inference is needed in the future, change this to the inference using SFTTrainer instead of pipeline (similar to module_classification)
        '''
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=self.dtype,
            tokenizer=self.tokenizer,
        )

        terminators = [pipeline.tokenizer.eos_token_id]
        if "Llama-3" in self.model_name: terminators.append(pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        responses = []
        prompts = [self.formatting_func_inference(question) for question in questions]
        
        # Use tqdm to show progress
        for prompt in tqdm(prompts, desc="Inferencing", unit="prompt"):
            result = pipeline(
                prompt,
                do_sample=False,
                num_return_sequences=1,
                return_full_text=False,
                eos_token_id=terminators,
                max_new_tokens=self.tokenizer_max_length,
            )
            responses.append(result[0]['generated_text'].lstrip())
            free_memory()
        return responses


    def formatting_dataset(self, examples, add_generation_prompt=False):
            '''
            Expects a list of example, each of form: {"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "..."}]}
            '''
            # apply_chat_template takes in: example['messages']
            texts = []
            for example in examples:
                texts.append(self.tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False))
            return Dataset.from_dict({'text' : texts})


    #Expects eval_data['text'] gives the texts (already formatted with correct template)
    def evaluate(self, eval_data, evaluate_output_dir, per_device_eval_batch_size=1):
        training_args = TrainingArguments(
            output_dir=evaluate_output_dir,  # where to save the results
            per_device_eval_batch_size=per_device_eval_batch_size,  # Adjust batch size according to your hardware capacity
            do_train=False,
            do_eval=True,
        )
        eval_trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            eval_dataset=eval_data,  # Supply the test data as eval_dataset
            tokenizer=self.tokenizer,
            packing=False,
            # formatting_func=lambda x: self.formatting_func(x),
            dataset_text_field="text",
            max_seq_length=self.tokenizer_max_length,
        )
        eval_results = eval_trainer.evaluate()
        print("Evaluation Results:", eval_results)

    #Expects train_data['text'] gives the texts (already formatted with correct template)
    def finetune(self, train_data, valid_data, lora_r, training_args, wandb_proj_name, save_model_path=""):
        '''
        Although it is claimed we could directly pass in OpenAI messages format (SFTTrainert will apply tokenizer.apply_chat_template method.):
            {"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "..."}]}
        But we will get the error below, because SFTTrainer uses .numpy and python so far does not support torch bfloat16:
        args = (obj.detach().cpu().numpy(),)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
        TypeError: Got unsupported ScalarType BFloat16

        Thus we decide to direct preprocess the data into Dataset_from_dict({'text' : texts}) format
        '''
        #wandb
        wandb.login(key=wandb_api_key, relogin=False)
        wandb.init(project=wandb_proj_name, config=training_args) # wandb.init(project="llama_finetuning", config=training_args)
        #data: we expected the caller to preprocess data and "train_data" here is already formatted with correct template!
        #lora
        if lora_r != -1: # lora_r==-1 means do not use lora.
            peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["gate_proj","down_proj","up_proj","q_proj","v_proj","k_proj","o_proj"],
            )
            self.model = get_peft_model(self.model, peft_config)
        #train
        if valid_data:
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=valid_data,
                max_seq_length=self.tokenizer_max_length, # default 1024, we specify expected maximum sequence
                tokenizer=self.tokenizer,
                packing=False,
                # formatting_func=lambda x: self.formatting_func(x),
                dataset_text_field="text",
                # callbacks=[LoggingCallback()],  # Include the custom logging callback
            )
        else: #valid_data is None
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_data,
                max_seq_length=self.tokenizer_max_length, # default 1024, we specify expected maximum sequence
                tokenizer=self.tokenizer,
                packing=False,
                # formatting_func=lambda x: self.formatting_func(x),
                dataset_text_field="text",
                # callbacks=[LoggingCallback()],  # Include the custom logging callback
            )
        free_memory()
        trainer.train()
        if lora_r != -1:
            self.model = self.model.merge_and_unload()
        #save model
        if save_model_path != "":
            print("Saving model...")
            
            self.model.save_pretrained(save_model_path)
            self.tokenizer.save_pretrained(save_model_path)
            print("Done! Saved model and tokenizer at:", save_model_path)

    def load_model(self, model_path):
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.to(self.device)
        #update self.model to be loaded model
        self.model = model
        self.tokenizer = tokenizer
        print("Done! Loaded model and tokenizer from:", model_path)

