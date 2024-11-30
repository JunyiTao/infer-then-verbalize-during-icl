## <p align="center">Inference and Verbalization Functions During In-Context Learning</p>

This repository contains the code for the paper "[Inference and Verbalization Functions During In-Context Learning]()". 

#### Create environment and install dependencies:
```
conda create -n infer-verbalize python=3.10
conda activate infer-verbalize
pip install -r requirements.txt
```


#### Evalute ICL performance:

- For NLI Datasets:
```
for t in "default" "ent_not_ent" "yes_no" "good_bad" "cat_dog" "foo_bar" "b_w";
do python main.py --dataset multi_nli --max_new_tokens 5 --n_examples=16 \
--model_name meta-llama/Meta-Llama-3.1-70B --label_mapping $t --batch_size 1 \
--n_trials=1 --prompt_template sentence --vllm;
done;
```

- For IMDb:
```
for t in "default" "good_bad" "true_false" "cat_dog" "foo_bar" "w_f";
do python main.py --dataset imdb --max_new_tokens 5 --n_examples=16 \
--model_name meta-llama/Meta-Llama-3.1-70B --label_mapping $t --batch_size 1 \
--n_trials=1 --prompt_template passage_sentiment --vllm;
done;
```

- For AGNews:
```
for t in "default" "good_bad" "true_false" "cat_dog" "foo_bar" "w_f";
do python main.py --dataset agnews_multi --max_new_tokens 5 --n_examples=32 \
--model_name meta-llama/Meta-Llama-3.1-70B --label_mapping $t --batch_size 1 \
--n_trials=1 --prompt_template text_linebreak --vllm;
done;
```

- For secondary tasks on MultiNLI:
```
for p in "nli_fic" "nli_length" "nli_negation" "nli_overlap" "nli_tele"
do python main.py --dataset multi_nli_ambi --max_new_tokens 5 --n_examples=16 \
--model_name meta-llama/Meta-Llama-3.1-70B --label_mapping default --batch_size 1 \
--ambi_pair $p --ambi_task t2 --demo_type disambi --always_inform --n_trials=1 \
--prompt_template ambi_instruct --vllm; 
done;
```

#### Replicate intervention results:

- For NLI Datasets (remove ``--save_output`` if you don't want to save the model outputs):
```
python intervene.py --dataset multi_nli --max_new_tokens 5 --n_examples=16 \
--model_name meta-llama/Meta-Llama-3.1-70B --n_trials 1 --seed 42 --do_label \
--batch_size 1 --prompt_template sentence --save_output
```

- For IMDb:
```
python intervene.py --dataset imdb --max_new_tokens 5 --n_examples=16 \
--model_name meta-llama/Meta-Llama-3.1-70B --n_trials 1 --seed 42 --do_label \
--batch_size 1 --prompt_template passage_sentiment --save_output
```

- For AGNews:
```
python intervene.py --dataset agnews_multi --max_new_tokens 5 --n_examples=32 \
--model_name meta-llama/Meta-Llama-3.1-70B --n_trials 1 --seed 42 --do_label \
--batch_size 1 --prompt_template text_linebreak --save_output
```

- For secondary tasks on MultiNLI:
```
python intervene.py --dataset multi_nli_ambi --max_new_tokens 5 --n_examples=16 \
--model_name meta-llama/Meta-Llama-3.1-70B --n_trials 1 --seed 42 --do_ambi \
--batch_size 1 --save_output --prompt_template ambi_instruct
```

