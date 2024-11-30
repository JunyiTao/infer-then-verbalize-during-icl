
import torch.utils
import torch.utils.data

import src.data as data
import src.model as model
import src.utils as utils
import torch
import argparse
from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
import json
import os
import random



INFO_MEM = None
TEST_MODEL = None


def yield_batch(iterable, batch_size):
    if type(iterable) == int:
        iterable = list(range(iterable))
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i+batch_size]



def eval_intervene_on_dataset(model: model.ICLModel, source_dataset: data.NLIDataset, intervene_dataset: data.NLIDataset, args,
                              batch_size=8, shuffle=False):
    
    global INFO_MEM


    
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=batch_size, shuffle=shuffle)
            
    intervene_loader = torch.utils.data.DataLoader(intervene_dataset, batch_size=batch_size, shuffle=False)
                                                

    if args.do_label:
        source_info = model.pred_and_save_hidden_states(source_loader, k=args.last_k, 
                                                        save_repr=True, 
                                                        cache_prefix=source_dataset.get_prompt() if args.use_cache else None
                                                        ) if INFO_MEM is None else INFO_MEM
        intervene_info = model.pred_and_save_hidden_states(intervene_loader,
                                                            cache_prefix=intervene_dataset.get_prompt() if args.use_cache else None,
                                                            return_cache=args.use_cache)
        if args.use_cache:
            intervene_info, cache_info = intervene_info
        INFO_MEM = source_info
    elif args.do_ambi:
        source_info = model.pred_and_save_hidden_states(source_loader, k=args.last_k, 
                                                        save_repr=True,
                                                        cache_prefix=source_dataset.get_prompt() if args.use_cache else None)
        intervene_info = model.pred_and_save_hidden_states(intervene_loader,
                                                            cache_prefix=intervene_dataset.get_prompt() if args.use_cache else None,
                                                            return_cache=args.use_cache
                                                            ) if INFO_MEM is None else INFO_MEM
        INFO_MEM = intervene_info
        if args.use_cache:
            intervene_info, cache_info = intervene_info

    
    source_pred = [i['pred'] for i in source_info]
    intervene_pred = [i['pred'] for i in intervene_info]

    source_pred_i = [source_dataset.inverse_answer(i) for i in source_pred]
    intervene_pred_i = [intervene_dataset.inverse_answer(i) for i in intervene_pred]
    
    i_inter = {label:[i for i in range(len(intervene_pred_i)) if intervene_pred_i[i] == label] for label in range(intervene_dataset.num_classes)}

    if sum(len(v) != 0 for v in i_inter.values()) <= 1:
        return None

    inverse_idx = []
    tmp = []
    for i in source_pred_i:
        if i in range(source_dataset.num_classes):
            to_sample = []
            for j in range(intervene_dataset.num_classes):
                if j != i:
                    to_sample.extend(i_inter[j])
            sample_idx = random.choice(to_sample)
            inverse_idx.append(sample_idx)
        else:
            to_sample = []
            for j in range(intervene_dataset.num_classes):
                to_sample.extend(i_inter[j])
            sample_idx = random.choice(to_sample)
            inverse_idx.append(sample_idx)
        tmp.append(intervene_pred[sample_idx])

        
    match_fn = source_dataset.get_cross_dataset_label_match_fn(intervene_dataset)
    match_acc = []


    if args.save_output:
        outputs = dict()



    for layer in range(model.num_layers - 1) : 
        layer_match = 0
        layer_total = 0
        if args.save_output:
            outputs[f'layer_{layer+1}'] = {'source_org_pred': [], 'intervened_pred': [], 'intervened_org_pred': [], 'matched': []}
        
        for source_idx, intervene_idx in zip(yield_batch(len(source_info), args.batch_size), yield_batch(inverse_idx, args.batch_size)):
            

            source_batch = [source_info[i] for i in source_idx]
            intervene_batch = [intervene_info[i] for i in intervene_idx]

            results = model.replace_last_repr([i['input'] for i in intervene_batch], [j['last_hs'][layer+1] for j in source_batch], 
                                              layer, intervene_target='layer', last_k=args.last_k, 
                                              cache=cache_info if args.use_cache else None)
            

            results = [[j['pred'] for j in source_batch],
                       results,
                       [j['pred'] for j in intervene_batch],
                       source_idx]
            if args.save_output:
                outputs[f'layer_{layer+1}']['source_org_pred'].extend(results[0])
                outputs[f'layer_{layer+1}']['intervened_pred'].extend(results[1])
                outputs[f'layer_{layer+1}']['intervened_org_pred'].extend(results[2])

            for tmp in zip(*results):
                org, new, intervened_org, source_id = tmp
                match = match_fn(org, new)
                layer_match += match
                if args.save_output:
                    outputs[f'layer_{layer+1}']['matched'].append(match)
                layer_total += 1
                
        
        match_acc.append(layer_match / layer_total)
    
    if args.save_output:
        return np.array(match_acc), outputs
    return np.array(match_acc)


def do_one_trial(args, seed, log_dir=None):
    global TEST_MODEL


    if not args.do_ambi:
        source_dataset = data.NLIDataset(dataset_name=args.dataset, n_examples=args.n_examples, prompt_template=args.prompt_template, 
                                    seed=seed, label_mapping=args.label_mapping, reverse=args.reverse, split='validation')
    else:
        source_dataset = data.NLIDataset(dataset_name=args.dataset, n_examples=args.n_examples, prompt_template=args.prompt_template, 
                        seed=seed, label_mapping=args.label_mapping, reverse=args.reverse, split='validation',
                        ambi_pair='nli_fic', ambi_task='nli', demo_type='disambi', use_inst=True, always_inform=True)
    
    intervene_datasets = dict()

    
    if args.do_label:

        for key in source_dataset.label_mapping_dict.keys():
        
            intervene_datasets[key] = data.NLIDataset(dataset_name=args.dataset, n_examples=args.n_examples, prompt_template=args.prompt_template, 
                                        seed=seed, label_mapping=key, reverse=args.reverse, split='validation')
                


        TEST_MODEL = model.ICLModel(source_dataset, model_name=args.model_name, intervene=True) if TEST_MODEL is None else TEST_MODEL
            

        results = dict()
        
        for key in tqdm(intervene_datasets.keys()):
            results[key] = eval_intervene_on_dataset(TEST_MODEL, source_dataset, intervene_datasets[key], args,
                                                        args.batch_size, shuffle=args.shuffle) # put representations from the default model into other label mappings
            
            if log_dir is not None and results[key] is not None:
                with open(f'{log_dir}/{key}.json', 'w') as f:
                    json.dump(results[key][1], f)

                                                      
    elif args.do_ambi:

        assert args.dataset == "multi_nli_ambi"

        
        intervene_datasets['nli'] = data.NLIDataset(dataset_name=args.dataset, n_examples=args.n_examples, prompt_template=args.prompt_template, 
                                            seed=seed, label_mapping=args.label_mapping, reverse=args.reverse, split='validation',
                                ambi_pair='nli_fic', ambi_task='nli', demo_type='disambi', use_inst=True, always_inform=True)
        
        intervene_datasets['fic'] = data.NLIDataset(dataset_name=args.dataset, n_examples=args.n_examples, prompt_template=args.prompt_template, 
                                            seed=seed, label_mapping=args.label_mapping, reverse=args.reverse, split='validation',
                                ambi_pair='nli_fic', ambi_task='t2', demo_type='disambi', use_inst=True, always_inform=True)
        
        intervene_datasets['tele'] = data.NLIDataset(dataset_name=args.dataset, n_examples=args.n_examples, prompt_template=args.prompt_template, 
                                            seed=seed, label_mapping=args.label_mapping, reverse=args.reverse, split='validation',
                                ambi_pair='nli_tele', ambi_task='t2', demo_type='disambi', use_inst=True, always_inform=True)
        
        intervene_datasets['negation'] = data.NLIDataset(dataset_name=args.dataset, n_examples=args.n_examples, prompt_template=args.prompt_template, 
                                            seed=seed, label_mapping=args.label_mapping, reverse=args.reverse, split='validation',
                                ambi_pair='nli_negation', ambi_task='t2', demo_type='disambi', use_inst=True, always_inform=True)

        intervene_datasets['overlap'] = data.NLIDataset(dataset_name=args.dataset, n_examples=args.n_examples, prompt_template=args.prompt_template, 
                                            seed=seed, label_mapping=args.label_mapping, reverse=args.reverse, split='validation',
                                ambi_pair='nli_overlap', ambi_task='t2', demo_type='disambi', use_inst=True, always_inform=True)
        
        TEST_MODEL = model.ICLModel(source_dataset, model_name=args.model_name, intervene=True) if TEST_MODEL is None else TEST_MODEL
            

        results = dict()
        for key in tqdm(intervene_datasets.keys()):
            results[key] = eval_intervene_on_dataset(TEST_MODEL, intervene_datasets[key], source_dataset, args,
                                                        args.batch_size, shuffle=args.shuffle) # put representations from other tasks into the default task
            
            if log_dir is not None and results[key] is not None:
                with open(f'{log_dir}/{key}.json', 'w') as f:
                    json.dump(results[key][1], f)

    else:
        raise ValueError("Invalid mode")
    
    to_return = dict()
    for key in results.keys():
        if results[key] is None:
            print(f"Skipping {key}")
        else:
            to_return[key] = results[key]
    return to_return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # model and dataset
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-14m")
    parser.add_argument("--dataset", type=str, default="rte")
    # settings
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_examples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--revision", type=str, default="main")
    # label mapping
    parser.add_argument("--label_mapping", type=str, default="default")
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument("--reverse", type=bool, default=False)
    # prompt template
    parser.add_argument("--prompt_template", type=str, default="sentence")
    # operations
    parser.add_argument("--do_label", action='store_true')
    parser.add_argument("--do_ambi", action='store_true')
    # save output
    parser.add_argument("--save_output", action='store_true')
    # more fine-grained target
    parser.add_argument("--last_k", type=int, default=1)

    parser.add_argument("--no_cache", action='store_true')



    args = parser.parse_args()

    args.use_cache = not args.no_cache


    if args.use_cache:
        assert args.batch_size == 1, "must have batch size 1 for caching"


    assert sum([args.do_label, args.do_ambi]) == 1, "Must specify exactly one of --do_label, or --do_ambi" 



    
    rand_int = random.randint(0, 100000)
    run_name = f'intervene-{args.model_name}-{args.dataset}-{args.prompt_template}-ex{args.n_examples}-{args.n_trials}trials-seed{args.seed}-{utils.get_hex_time(ms=True)}-{rand_int}'

    run_name = run_name.replace('/', '_')

    
    trial_results = []
    for trial in range(args.n_trials):
        torch.cuda.empty_cache()
        INFO_MEM = None
        seed = args.seed + trial

        if args.save_output:
            log_dir = f'output/{run_name}/trial_{trial}'
                
            os.makedirs(f'{log_dir}', exist_ok=True)
        
        curves = do_one_trial(args, seed, log_dir if args.save_output else None)
        if args.save_output:
            curves = {k: v[0] for k,v in curves.items()}
        trial_results.append(curves)

        
    curves = {k: np.mean([result[k] for result in trial_results if k in result],axis=0) for k in trial_results[0].keys()}
    curves_std = {k: np.std([result[k] for result in trial_results if k in result],axis=0) for k in trial_results[0].keys()}



    # make a table where each row is a curve
    # each column is a layer
    df = pd.DataFrame(curves, columns=curves.keys())
    df_std = pd.DataFrame(curves_std, columns=curves.keys())

    model_name = args.model_name.replace('/', '_')
    os.makedirs(f'results/{model_name}-{args.dataset}-{args.seed}-{args.n_trials}trials/', exist_ok=True)
    df.to_csv(f'results/{model_name}-{args.dataset}-{args.seed}-{args.n_trials}trials/mean.csv')
    df_std.to_csv(f'results/{model_name}-{args.dataset}-{args.seed}-{args.n_trials}trials/std.csv')

    
