import torch
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict, concatenate_datasets, ClassLabel
import datasets
import random
import os
import pandas as pd
import json
import numpy as np
import re

TEST_SIZE = 300
        
    
class NLIDataset(Dataset):
    def __init__(self, dataset_name="rte", n_examples=8, seed=42, label_mapping='default', reverse=False, shuffle=False,
                 prompt_template='sentence', subset=1.0, split='validation', ambi_pair='reuters_2', ambi_task='keyword', demo_type='ambi', 
                 use_inst=True, always_inform=False):
        
        
        self.n_examples = n_examples

        self.name = dataset_name.lower()
        assert self.name in ["rte", "anli","imdb","agnews_multi",'multi_nli','multi_nli_ambi'], f"Dataset {self.name} not supported."
        if self.name == "rte":
            self.dataset = load_dataset("super_glue", "rte", trust_remote_code=True)
            self.premise_key = "premise"
            self.hypothesis_key = "hypothesis"
            self.label_key = "label"

        elif self.name == "imdb":
            self.dataset = load_dataset("imdb", trust_remote_code=True)
            val_set = self.dataset['test'].train_test_split(test_size=TEST_SIZE, seed=4869, stratify_by_column='label')['test']
            self.dataset = DatasetDict({'train': self.dataset['train'].shard(10, 0), 
                                        'validation': val_set})
            self.premise_key = "text"
            self.hypothesis_key = "text"
            self.label_key = "label"
            def process_ex(x):
                x['label'] = 1 - x['label']
                x['text'] = x['text'][:min(350, len(x['text']))]
                return x
            
            self.dataset = self.dataset.map(process_ex)
            
        elif self.name == "anli":
            
            self.dataset = load_dataset("anli", trust_remote_code=True)
            self.dataset = self.dataset.filter(lambda example: example["label"] != 1)
            def process_ex(x):
                x['label'] = 0 if x['label'] == 0 else 1
                return x
            self.dataset = self.dataset.map(process_ex)
            val_set = self.dataset['dev_r1'].train_test_split(test_size=TEST_SIZE, seed=4869, stratify_by_column='label')['test']
            self.dataset = DatasetDict({'train': self.dataset['train_r1'].shard(6, 0), 
                                        'validation': val_set})
            self.premise_key = "premise"
            self.hypothesis_key = "hypothesis"
            self.label_key = "label"
            self.dataset.select_columns(['premise', 'hypothesis', 'label'])


        elif self.name == "multi_nli":
            
            self.dataset = load_dataset("nyu-mll/multi_nli", trust_remote_code=True)
            self.dataset = self.dataset.filter(lambda example: example["label"] != 1)
            def process_ex(x):
                x['label'] = 0 if x['label'] == 0 else 1
                return x
            self.dataset = self.dataset.map(process_ex)
            val_set = self.dataset['validation_matched'].train_test_split(test_size=TEST_SIZE, seed=4869, stratify_by_column='label')['test']
            self.dataset = DatasetDict({'train': self.dataset['train'].shard(157, 0), 
                                        'validation': val_set})
            self.premise_key = "premise"
            self.hypothesis_key = "hypothesis"
            self.label_key = "label"
            self.dataset.select_columns(['premise', 'hypothesis', 'label'])


        
        elif self.name == 'multi_nli_ambi':
            t1, t2 = ambi_pair.split('_')
            assert t1 == 'nli', 'Only nli is supported on multi_nli'
            assert t2 in ['fic', 'length', 'negation', 'overlap' , 'tele'], f'ambiguity task {t2} is not supported'
            assert demo_type in ['ambi', 'disambi'], f'disambiguation type {demo_type} is not supported'

            self.dataset, sample_idx = self.construct_amibiguous_dataset_multinli(ambi_pair, demo_type, seed=seed)

            self.premise_key = "premise"
            self.hypothesis_key = "hypothesis"
            if ambi_task == 'nli':
                self.label_key = "nli_label"
            elif ambi_task == 't2':
                self.label_key = "t2_label"
            else:
                raise NotImplementedError("ambi_task should be either 'nli' or 't2'.")

            self.use_inst = use_inst
            if self.use_inst:
                with open('src/prompt_template/multi_nli_ambi_inst.json', 'r') as f:
                    insts = json.load(f)
                if demo_type == 'ambi' or always_inform:
                    self.inst = insts[t2] if ambi_task == 't2' else insts['nli']
                else:
                    self.inst = insts['uninformative']

        elif self.name == 'agnews_multi':
            self.dataset = load_dataset('ag_news', trust_remote_code=True)
            self.premise_key = "text"
            self.hypothesis_key = "text"
            self.label_key = "label"

            val_set = self.dataset['test'].train_test_split(test_size=TEST_SIZE, seed=4869, stratify_by_column='label')['test']
            self.dataset = DatasetDict({'train': self.dataset['train'].shard(48, 0), 
                                        'validation': val_set})

            assert self.n_examples % 4 == 0, "n_examples should be a multiple of 4."
            sample_idx = []
            random.seed(seed)
            for i in range(4):
                type_idx = [j for j, x in enumerate(self.dataset['train']['label']) if x == i]
                sample_idx += random.sample(type_idx, self.n_examples // 4)
            random.shuffle(sample_idx)
            

        else:
            self.dataset = load_dataset(self.name, trust_remote_code=True)
            raise NotImplementedError("Only RTE is supported for now.")

        random.seed(seed)
        

        # balanced sampling
        if self.name not in ["agnews", "multi_nli_ambi", "agnews_multi"]:
            assert self.n_examples % 2 == 0, "n_examples should be a multiple of 2."                                                             
            sample_idx = []                                                                                                                  
            for i in range(2):                                                                                                                       
                type_idx = [j for j, x in enumerate(self.dataset['train']['label']) if x == i]
                sample_idx += random.sample(type_idx, self.n_examples // 2)                                                                      
            random.shuffle(sample_idx) 

        self.examples = [self.dataset['train'][i] for i in sample_idx]
        
        self.shuffle = shuffle
        self.seed = seed

        if split == 'validation':
            self.test_data = self.dataset['validation']
        elif split =='train':
            self.test_data = self.dataset['train'].filter(lambda example, idx: idx not in sample_idx, with_indices=True)

        if subset < 1.0:
            self.test_data = self.test_data.select(range(int(subset * self.test_data.num_rows)))

        self.label_mapping = label_mapping
        self.reverse = reverse
        self.spliter = '\n\n'


        # should average across templates of the same category
        with open("src/prompt_template/templates.json", 'r') as file:
            templates_dict = json.load(file)
        
        self.prompt_templates_dict = templates_dict

        if prompt_template in templates_dict:
            # Set the prompt template
            self.prompt_template = templates_dict[prompt_template]['template']
        else:
            # Handle the case where the template name is not found
            raise ValueError(f"Template '{prompt_template}' not found.")


        if self.name == "strategyqa":
            with open('src/prompt_template/strategyqa_prompt.txt', 'r') as f:
                self.prompt_template = f.read()
        

        if self.name not in ["agnews_multi", "imdb"]: # nli datasets
            self.label_mapping_dict = {
            'default': ['true', 'false'],
            'ent_not_ent': ['entailment', 'not_entailment'],
            'yes_no': ['yes', 'no'],
            'good_bad': ['good', 'bad'],
            'cat_dog': ['cat', 'dog'],
            'foo_bar': ['foo', 'bar'],
            'b_w': ['black', 'white']
        }
        
            
            self.label_mapping_func = lambda x: self.label_mapping_dict[self.label_mapping][x]
            if prompt_template in ['imply_inst_token', 'imply_inst_token_no_hit']:
                self.prompt_template = self.prompt_template.format(l0=self.label_mapping_func(0), l1=self.label_mapping_func(1),
                                                               hypothesis='{hypothesis}', premise='{premise}', answer='{answer}')

        elif self.name == "imdb":
            self.label_mapping_dict = {
            'default': ['positive', 'negative'],
            'good_bad': ['good', 'bad'],
            'true_false': ['true', 'false'],
            "cat_dog": ["cat", "dog"],
            "foo_bar": ["foo", "bar"],
            'w_f': ['water', 'fire']
        }
            assert self.label_mapping in self.label_mapping_dict, f"Label mapping {self.label_mapping} not supported."
            
            self.label_mapping_func = lambda x: self.label_mapping_dict[self.label_mapping][x]

        elif self.name == "agnews_multi":
            self.label_mapping_dict = {
            'default': ['world', 'sports', 'business', 'sci/tech'],
            'topic_re': ['international', 'athletics', 'commerce', 'innovation'],
            'letter': ['a', 'b', 'c', 'd'],
            'number': ['1', '2', '3', '4'],
            'foo_bar': ['foo', 'bar', 'baz', 'qux'],
        }
            assert self.label_mapping in self.label_mapping_dict, f"Label mapping {self.label_mapping} not supported."
            self.label_mapping_func = lambda x: self.label_mapping_dict[self.label_mapping][x]
            self.prompt_template = self.prompt_template.format(l1=self.label_mapping_func(0), l2=self.label_mapping_func(1), 
                                                               l3=self.label_mapping_func(2), l4=self.label_mapping_func(3),
                                                               premise='{premise}', answer='{answer}')
        

        self.num_classes = 2 if self.name != "agnews_multi" else 4

    def __len__(self):
        return self.test_data.num_rows
    


    def construct_amibiguous_dataset_multinli(self, ambi_pair, demo_type, seed=42):


        assert self.n_examples % 4 == 0, "n_examples should be a multiple of 4."
        assert self.n_examples <= 40, "n_examples should be less than 40."


        _, t2 = ambi_pair.split('_')
        
        with open(f'src/dataset/nli_entailment_{t2}.json') as f:
            dataset = json.load(f)
        
        

        def build_from_split(split, ambi_type):
            result = {'premise':[], 'hypothesis':[], 'label':[], 'nli_label':[], 't2_label':[], 'ambi_type':[]}
            for sample in split:
                p, h = sample['question'].split('\n')
                result['premise'].append(p)
                result['hypothesis'].append(h)
                nli_label, t2_label = sample['split'].split('_')
                result['nli_label'].append(1 - int(nli_label))
                result['t2_label'].append(1 - int(t2_label))
                result['label'].append(1 - int(nli_label))
                result['ambi_type'].append(ambi_type)
            return datasets.Dataset.from_dict(result)
        

        both = build_from_split(dataset['demos_1_1']+dataset['testset_1_1'], 0)
        t2_only = build_from_split(dataset['testset_0_1'], 1)
        ent_only = build_from_split(dataset['testset_1_0'], 2)
        neither = build_from_split(dataset['demos_0_0']+dataset['testset_0_0'], 3)

        
        
        testset_size = TEST_SIZE

        min_len = min([len(i)-(testset_size // 4) for i in [both, t2_only, ent_only, neither]])

        train_size = min_len * 4
        total_size = train_size + testset_size
        

        random.seed(seed)
        data_idx = random.sample(range(both.num_rows), total_size // 4)
        both = both.select(data_idx)
        data_idx = random.sample(range(t2_only.num_rows), total_size // 4)
        t2_only = t2_only.select(data_idx)
        data_idx = random.sample(range(ent_only.num_rows), total_size // 4)
        ent_only = ent_only.select(data_idx)
        data_idx = random.sample(range(neither.num_rows), total_size // 4)
        neither = neither.select(data_idx)
        
        dataset = concatenate_datasets([both, t2_only, ent_only, neither])

        new_features = dataset.features.copy()
        new_features["ambi_type"] = ClassLabel(names=["both", "t2_only", "ent_only", "neither"])
        dataset = dataset.cast(new_features)
        dataset = dataset.train_test_split(test_size=testset_size, seed=seed, stratify_by_column='ambi_type')
        dataset['validation'] = dataset['test']

        sample_idx = []
        if demo_type == 'ambi':
            for i in [0, 3]:
                type_idx = [j for j, x in enumerate(dataset['train']['ambi_type']) if x == i]
                sample_idx += random.sample(type_idx, self.n_examples // 2)
        elif demo_type == 'disambi':
            for i in range(4):
                type_idx = [j for j, x in enumerate(dataset['train']['ambi_type']) if x == i]
                sample_idx += random.sample(type_idx, self.n_examples // 4)
        
        random.shuffle(sample_idx)
        
        
        return dataset, sample_idx



    def inverse_answer(self, label):
        partial = self.label_mapping_dict[self.label_mapping]
        partial = [i.startswith(label) for i in partial]
        try:
            assert sum(partial) == 1
            return partial.index(True)
        except:
            return -1
        
    def get_answer_format(self, label, shuffle=False, demo_idx=0):
        if self.reverse:
            label = 1 - label

        if shuffle:
            random.seed(self.seed + demo_idx)
            label = random.choice([0, 1])

        return self.label_mapping_func(label)


        
    def get_cross_dataset_label_match_fn(self, other_dataset):
        
        def match_fn(self_label, other_label):
            self_mapping = self.label_mapping_dict[self.label_mapping]
            other_mapping = other_dataset.label_mapping_dict[other_dataset.label_mapping]

            self_mapping = [i.startswith(self_label) for i in self_mapping]
            other_mapping = [i.startswith(other_label) for i in other_mapping]

            if True not in self_mapping or True not in other_mapping:
                return False
            return self_mapping.index(True) == other_mapping.index(True)

        return match_fn
    
    def get_prompt(self, test_example=None):
        prompt = ''
        if self.name == 'agnews':
            prompt = f'Output {self.label_mapping_func(0)} if [condition withheld] otherwise {self.label_mapping_func(1)}.\n'
            
        for i, demo in enumerate(self.examples):
            if self.name == 'multi_nli_ambi' and self.use_inst:
                prompt += self.inst.format(pos_label=self.label_mapping_func(0), neg_label=self.label_mapping_func(1)) + '\n'
            prompt += self.prompt_template.format(premise=demo[self.premise_key].strip(), 
                                            hypothesis=demo[self.hypothesis_key].strip(), 
                                            answer=self.get_answer_format(demo[self.label_key], 
                                                                          shuffle=self.shuffle, demo_idx=i))
            prompt += self.spliter
        
        if test_example is not None:    
            if self.name == 'multi_nli_ambi' and self.use_inst:
                prompt += self.inst.format(pos_label=self.label_mapping_func(0), neg_label=self.label_mapping_func(1)) + '\n'
            prompt += self.prompt_template.format(premise=test_example[self.premise_key].strip(), 
                                        hypothesis=test_example[self.hypothesis_key].strip(), 
                                        answer='')
        
        return prompt
    
    def sample_by_label(self, label):
        return random.choice([i for i in self if i['label'] == label])
    
    def __getitem__(self, idx):
        result = {'input': self.get_prompt(self.test_data[idx])}
        result.update(self.test_data[idx])
        return result
        

