import src.data as data
import src.model as model
import torch
import argparse
import numpy as np
from collections import Counter

TEST_MODEL = None


def do_one_trial(args, seed):
    global TEST_MODEL
    if args.dataset in ['rte', 'anli','imdb', 'agnews_multi', 'multi_nli','multi_nli_ambi']:
        testdata = data.NLIDataset(dataset_name=args.dataset, n_examples=args.n_examples, seed=seed, prompt_template=args.prompt_template,
                                   label_mapping=args.label_mapping, shuffle=args.shuffle, reverse=args.reverse,
                                   ambi_pair=args.ambi_pair, ambi_task=args.ambi_task, demo_type=args.demo_type, use_inst=not args.no_inst,
                                   always_inform=args.always_inform)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    print(testdata[0]['input'])

    test_loader = torch.utils.data.DataLoader(testdata, batch_size=args.batch_size, shuffle=True)
    TEST_MODEL = model.ICLModel(testdata, model_name=args.model_name, revision=args.revision, use_vllm=args.vllm) if TEST_MODEL is None else TEST_MODEL
    acc, y_list, y_hat_list = TEST_MODEL.evaluate(test_loader, max_new_tokens=args.max_new_tokens, check_contain=args.check_contain)

    return acc, y_hat_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # model and dataset
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--dataset", type=str, default="rte")
    # settings
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_examples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--revision", type=str, default="main")
    # label mapping
    parser.add_argument("--label_mapping", type=str, default="default")
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument("--reverse", action='store_true')
    # prompt template
    parser.add_argument("--prompt_template", type=str, default="sentence")
    # others
    parser.add_argument("--check_contain", action='store_true')
    parser.add_argument("--ambi_pair", type=str, default="reuters_2")
    parser.add_argument("--ambi_task", type=str, default="keyword")
    parser.add_argument("--demo_type", type=str, default="ambi")
    parser.add_argument("--no_inst", action='store_true')
    parser.add_argument("--always_inform", action='store_true')
    parser.add_argument("--vllm", action='store_true')


    args = parser.parse_args()


    if args.label_mapping == 'shuffle':
        args.shuffle = True
        args.label_mapping = 'default'

    if args.label_mapping == 'reverse':
        args.reverse = True
        args.label_mapping = 'default'


    trial_results = []
    output_counter = Counter()
    for i in range(args.n_trials):
        torch.cuda.empty_cache()
        acc, y_hat_list = do_one_trial(args, args.seed+i)
        trial_results.append(acc)
        output_counter.update(y_hat_list)

    acc = np.mean(trial_results)
    std = np.std(trial_results)


    
    print(f'Acuuracy: {acc:.2f} +/- {std:.2f}')

    print(f'Accuracy per run: {trial_results}')

