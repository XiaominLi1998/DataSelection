from modules.utils import *

class LLMTextClassification:
    def __init__(self, model_name, num_labels, tokenizer_max_length=128, dtype_str='bfloat16', device='cuda'):
        # model_name = add_path_prefix(model_name)
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer_max_length = tokenizer_max_length
        self.saved_model_path = None
        # Initialize the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.device = device
        self.dtype = get_torch_dtype(dtype_str)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, token=hf_token, torch_dtype=self.dtype)
        # if 'llama' in self.model_name.lower():
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)


    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy, precision, recall, f1, mcc = classification_loss(predictions, labels)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
        }

    def inference(self, questions, inference_output_dir):
        inference_data = CustomDataset_Inference(questions, self.tokenizer, tokenizer_max_length=self.tokenizer_max_length)
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=inference_output_dir,
                per_device_eval_batch_size=1
            )
        )
        predictions = trainer.predict(inference_data)
        logits = predictions.predictions
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
        predicted_labels = np.argmax(probabilities, axis=1)
        print("Probabilities:", probabilities)
        print("Predicted labels:", predicted_labels)
        return predicted_labels
        # XM: Similar to evaluate: Original method below using pipeline is slow. We use trainer.predict above.
        # return query(self.model, self.tokenizer, questions, self.dtype)

    def evaluate(self, eval_dataset, evaluate_output_dir, per_device_eval_batch_size=1):
        eval_data = CustomDataset_Train_Cls(list(eval_dataset['text']), list(eval_dataset['label']), self.tokenizer, tokenizer_max_length=self.tokenizer_max_length)
        eval_training_args = TrainingArguments(
            output_dir=evaluate_output_dir,  # where to save the results
            per_device_eval_batch_size=per_device_eval_batch_size,
            do_train=False,
            do_eval=True,
            report_to=[]
        )
        eval_trainer = Trainer(
            model=self.model,
            args=eval_training_args,
            eval_dataset=eval_data,
            compute_metrics=self.compute_metrics,
        )

        eval_results = eval_trainer.evaluate()
        print("Evaluation Results:", eval_results)
        # XM: Original method below using pipeline results + loss calculation is slow. We use Trainer evaluation above.
        # results = self.inference(eval_dataset['text'])
        # predictions = [int(res['label'][6:]) for res in results]
        # accuracy, precision, recall, f1 = classification_loss(predictions, eval_dataset['label'])
        # print(f"Accuracy:{accuracy}")
        # print(f"Precision:{precision}")
        # print(f"Recall:{recall}")
        # print(f"F1:{f1}")


    #Expects train_dataset['text'] gives the texts and ['label'] gives labels (for example it could be pandas dataframe)
    # lora_r==-1 means do not use lora.
    def finetune(self, train_dataset, valid_dataset, lora_r, training_args, wandb_proj_name, save_model_path=""):
        #wandb
        wandb.login(key=wandb_api_key, relogin=False)
        wandb.init(project=wandb_proj_name, config=training_args) # wandb.init(project="llama_finetuning", config=training_args)
        #data:
        train_data = CustomDataset_Train_Cls(list(train_dataset['text']), list(train_dataset['label']), self.tokenizer, tokenizer_max_length=self.tokenizer_max_length)

        #lora
        if lora_r != -1:
            peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=lora_r,
                bias="none",
                task_type="SEQ_CLS",
                target_modules=["gate_proj","down_proj","up_proj","q_proj","v_proj","k_proj","o_proj"],
            )
            self.model = get_peft_model(self.model, peft_config)
        #train
        if valid_dataset != None:
            valid_data = CustomDataset_Train_Cls(list(valid_dataset['text']), list(valid_dataset['label']), self.tokenizer, tokenizer_max_length=self.tokenizer_max_length)
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=valid_data,
                compute_metrics=self.compute_metrics,
            )
        else:
             trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_data,
                compute_metrics=self.compute_metrics,
            )
        
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
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.to(self.device)
        #update self.model to be loaded model
        self.model = model
        self.tokenizer = tokenizer
        print("Done! Loaded model and tokenizer from:", model_path)

