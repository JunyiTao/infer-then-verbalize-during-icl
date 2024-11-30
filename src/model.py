import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.autonotebook import tqdm
import numpy as np
from tqdm.autonotebook import tqdm
import pickle
import os
try:
    import vllm
except ImportError:
    print("VLLM not installed")
    pass




class ICLModel:
    
    def __init__(self, dataset, model_name="EleutherAI/pythia-70m-deduped", revision="main", use_vllm=False, intervene=False):

        if use_vllm:
            self.model = vllm.LLM(model=model_name, revision=revision,
                                  dtype="auto",
                                  trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=intervene, max_model_len=8192)
            self.num_layers = len(self.model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers)
            print("Using VLLM")
        else:
            self.get_model(model_name, revision)

        self.use_vllm = use_vllm
        self.label_key = dataset.label_key
        self.get_answer_format = dataset.get_answer_format
        self.dataset = dataset
        


    def get_model(self, model_name, revision):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name == "google/gemma-2-27b":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, revision=revision, device_map='auto', 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, revision=revision, device_map='auto', 
                torch_dtype="auto", 
                trust_remote_code=True)
        self.model = self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers
        print(next(self.model.parameters()).dtype)
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        try:
            self.maxlen = self.model.config.max_position_embeddings
        except AttributeError:
            self.maxlen = self.tokenizer.model_max_length
        


    @property
    def device(self):
        return next(self.model.parameters()).device
    

    def generate(self, prompt: list[str], new_token_only=False, **kwargs):
        input_info = self.tokenizer(prompt, return_tensors="pt", padding=True, 
                                   truncation=True, max_length=self.maxlen)
        
        input_info = {k: v.to(self.device) for k, v in input_info.items()}

        output = self.model.generate(**input_info, pad_token_id=self.tokenizer.eos_token_id, num_beams=1, do_sample=False, **kwargs)
        if new_token_only:
            output = output[:, input_info['input_ids'].shape[1]:]
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)
    
    def generate_vllm(self, prompt: list[str], **kwargs):
        params = vllm.SamplingParams(n=1, top_k=1, max_tokens=kwargs.get('max_new_tokens', 2))
        outputs = self.model.generate(prompt, sampling_params=params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]
        
    

    def replace_last_repr(self, intervene_prompt: list[str], source_repr: list[torch.Tensor], layer, last_k=1,
                          intervene_target='layer', cache=None):
        
        
        # Reshape source_repr to [:,k,:]
        reshaped_source_repr = [tensor.view(1, -1, tensor.size(-1)) for tensor in source_repr]
        
        # Stack the padded tensors
        stacked_source_repr = torch.cat(reshaped_source_repr, dim=0)


        def hook_fn(module, _input, output):

            to_replace = output if intervene_target == 'mlp' else output[0] 

            stacked_source_repr.to(to_replace.device)
            
            to_replace[:, -last_k:, :] = stacked_source_repr[:, -last_k:, :]


        with torch.no_grad():
            hook = self.model.model.layers[layer].register_forward_hook(hook_fn)
            replace_hidden_states = self.run_and_get_hidden_states(intervene_prompt, 
                                                                    past_key_values=cache[0] if cache is not None else None, 
                                                                    cache_len=cache[1] if cache is not None else None,
                                                                    )
            hook.remove()
            # to_return = to_return[-1][:,-1,:]
            replace_output_logit = self.model.lm_head(replace_hidden_states[-1][:,-1,:])
            replace_output_token = self.tokenizer.batch_decode(replace_output_logit.argmax(dim=-1), skip_special_tokens=True)

        stacked_source_repr.to('cpu')

        return replace_output_token
    



    
    def run_and_get_hidden_states(self, prompt: list[str], cache=False, past_key_values=None, cache_len=None, **kwargs):
        self.model.eval()
        input_info = self.tokenizer(prompt, return_tensors="pt", padding=True, 
                                   truncation=True, max_length=self.maxlen)
        
        input_info = {k: v.to(self.device) for k, v in input_info.items()}


        if past_key_values is not None:
            # remove first cache_len tokens
            input_info['input_ids'] = input_info['input_ids'][:, cache_len:]


            batch_size, seq_length = input_info['input_ids'].shape
            position_ids = torch.arange(cache_len, cache_len + seq_length).unsqueeze(0).repeat(batch_size, 1)
            position_ids = position_ids.to(self.device)

            # Repeat past_key_values for each batch element
            past_key_values = tuple(
                tuple(past_key_value.expand(batch_size, -1, -1, -1) for past_key_value in layer)
                for layer in past_key_values
            )
            with torch.no_grad():
                output = self.model(input_ids=input_info['input_ids'], return_dict=True, 
                                    output_hidden_states=True,
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    position_ids=position_ids,
                                    **kwargs)
        else:
            with torch.no_grad():
                output = self.model(**input_info, return_dict=True, 
                                    output_hidden_states=True,
                                    use_cache=cache,
                                    **kwargs)

        if cache:
            return output.past_key_values, input_info['input_ids'].shape[1]
        return output.hidden_states
    
    def pred_and_save_hidden_states(self, dataloader, cache=None, k=1, save_repr=False, cache_prefix=None, return_cache=False, **kwargs):


        self.model.eval()
        info = []
        num_batches = len(dataloader)

   
        if cache_prefix is not None:
            cache, cache_len = self.run_and_get_hidden_states(cache_prefix, cache=True)
            
            




        for batch in tqdm(dataloader):
            prompt = batch['input']

            if cache_prefix is not None:
                hs = self.run_and_get_hidden_states(prompt, past_key_values=cache, cache_len=cache_len, **kwargs)
            else:
                hs = self.run_and_get_hidden_states(prompt, **kwargs)
            

            pred = self.model.lm_head(hs[-1][:,-1,:])
            pred = self.tokenizer.batch_decode(pred.argmax(dim=-1), skip_special_tokens=True)
            
            for i in range(len(prompt)):
                if save_repr:
                    info.append({
                        'pred': pred[i],
                        'last_hs': [j[i, -k:, :].cpu() for j in hs],
                        'input': prompt[i]
                    })
                else:
                    info.append({
                        'pred': pred[i],
                        'input': prompt[i]
                    })


        if return_cache:
            return info, (cache, cache_len)
        return info
        

    

    def evaluate_batch(self, prompt: list[str], labels: list[str], get_answer_format=None, check_contain=False, verbose=True, **kwargs):
        if self.use_vllm:
            pred = self.generate_vllm(prompt, **kwargs)
        else:
            self.model.eval()
            with torch.no_grad():
                pred = self.generate(prompt, new_token_only=True, **kwargs)

        correct = 0
        y_list_batch, y_hat_list_batch = [], []
        for x, y_hat, y in zip(prompt, pred, labels):

            if verbose:
                print('-'*20)  
                print(y_hat)

            if '\n' in y_hat and not check_contain:
                y_hat = y_hat[:y_hat.index('\n')]
            y_hat = y_hat.rstrip().lower()

            if get_answer_format is not None:
                y = get_answer_format(y)
            else:
                y = self.get_answer_format(y)

            question = x.split(self.dataset.spliter)[-1].strip()

            if not check_contain:
                is_correct = y_hat == y
            else:
                is_correct = y.lower() in y_hat
                is_correct = is_correct and ('true' not in y_hat or 'false' not in y_hat)

            if len(question) <= 5:
                print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
                print(x)
                print(x.split(self.dataset.spliter))
                print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
            
            if verbose: 
                print(f'\n*************\nTemplate [{self.dataset.prompt_template}] Label [{y}] Pred [{y_hat}] -> Match [{is_correct}]\nQuestion: {question}')
                #   \nPredicted: {y_hat}, Actual: {y}\n')
            if is_correct:
                correct += 1
            y_list_batch.append(y)
            y_hat_list_batch.append(y_hat)
        return correct, y_list_batch, y_hat_list_batch


    def evaluate(self, dataloader, check_contain=False, **kwargs):
        if not self.use_vllm:
            self.model.eval()
        correct_count = 0
        y_list, y_hat_list = [], []
        total = 0
        for batch in tqdm(dataloader):
            prompt, labels = batch['input'], batch[self.label_key]
            correct, y_list_batch, y_hat_list_batch = self.evaluate_batch(prompt, labels, check_contain=check_contain, **kwargs)
            correct_count += correct
            y_list.extend(y_list_batch)
            y_hat_list.extend(y_hat_list_batch)
            total += len(prompt)
        return correct_count / total, y_list, y_hat_list
        
