# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py
import re
import os
import json
import random
import torch
import numpy as np
import transformers
from tqdm import tqdm, trange
import argparse
from collections import defaultdict, Counter
import glob
import sys
import time
import ssl
import urllib.request
import zipfile
import sys
from datasets import load_dataset,load_from_disk
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from decoding_algorithm import ContrastiveDecoding

transformers.logging.set_verbosity(40)


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

# N_SHOT = 7
# COT_FLAG = True
# ANSWER_TRIGGER = "So the answer is"

def load_csv(dataset_name, debug):
    # input file is in csv format, can be loaded by pandas
    # required columns: [prompt] only

    if dataset_name == 'triviaqa':
        dataset = load_from_disk('/app/data/ICD-main-1/datasets/trivia_qa')['validation']
    elif dataset_name == 'natural_questions':
        dataset = load_from_disk("/app/data/ICD-main-1/datasets/natural_questions")['validation']
    elif dataset_name == 'hotpotqa':
        dataset = load_dataset("hotpotqa")['validation']
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    list_data = list(dataset['question'])
    labels = list(dataset['answer'])

    if debug:
        list_data = list_data[0:20]
        labels = labels[0:20]

    return list_data, labels


def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = answer
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def build_prompt(question_text, prompt_style='zero_shot'):
    # this prompt is designed for trivia QA
    if prompt_style == 'zero_shot':
        question_text_prompt = 'Answer the following question concisely.\n'
        question_text_prompt += f'Q:{question_text}\nA:'
    elif prompt_style == 'few_shot':
        question_text_prompt = 'Answer the following question concisely.\n'
        question_text_prompt += f'Q: Who was President when the first Peanuts cartoon was published?\nA: Harry Truman\n\n'
        # question_text_prompt += f'Q: Which American-born Sinclair won the Nobel Prize for Literature in 1930?\nA: Sinclair Lewis\n\n'
        question_text_prompt += f'Q: Where in England was Dame Judi Dench born?\nA: York\n\n'
        question_text_prompt += f'Q: {question_text}\nA: '
    elif prompt_style == 'zero_shot_w_instru':
        raise NotImplementedError("zero_shot_w_instru Not implemented yet.")
    return question_text_prompt


def plot_auroc_scores(is_correct_list, scores_list, output_file, method_name):
    # Separate scores into correct and incorrect
    correct_scores = [score for is_correct, score in zip(is_correct_list, scores_list) if is_correct]
    incorrect_scores = [score for is_correct, score in zip(is_correct_list, scores_list) if not is_correct]

    # check if correct_scores and incorrect_scores are nan
    if np.isnan(correct_scores).any() or np.isnan(incorrect_scores).any():
        print(f"Error: there is nan, skip computing AUROC, AUPRC, AURC for {method_name}")
        auroc = None
        auprc = None
        aurc = None
        scores = {'auroc': auroc, 'auprc': auprc, 'aurc': aurc}
        return scores

    y_true = [1] * len(correct_scores) + [0] * len(incorrect_scores)
    y_scores = correct_scores + incorrect_scores

    # Compute AUROC
    auroc = roc_auc_score(y_true, y_scores)

    # Compute AUPRC
    auprc = average_precision_score(y_true, y_scores)

    # Compute AURC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    aurc = auc(recall, precision)

    # Create the plot
    plt.figure()
    plt.hist(correct_scores, bins=20, alpha=0.5, label='Correct')
    plt.hist(incorrect_scores, bins=20, alpha=0.5, label='Incorrect')
    plt.legend(loc='upper right')
    plt.title(f'AUROC: {auroc:.2f}')

    # Save the plot
    output_dir = os.path.dirname(output_file)
    plt.savefig(os.path.join(output_dir, f'detect_{method_name}_plot.png'))
    plt.close()

    scores = {'auroc': auroc, 'auprc': auprc, 'aurc': aurc}
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/app/data/llama2-7b-chat-hf")
    parser.add_argument("--amateur_model_name", type=str, default="/app/data/llama_epsilon_0.005_step_0.00005_steps_3_ratio1")
#    parser.add_argument("--amateur_model_name", type=str, default="/app/data/llama_random")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--mode", type=str, default="baseline")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--output-path", type=str, default="trivia.json")
    parser.add_argument("--prompt_style", type=str, choices=["zero_shot", "few_shot", "zero_shot_w_instru"], default='few_shot')
    # following four parameters are added
    parser.add_argument("--dataset_name", type=str, choices=["triviaqa", "natural_questions", "hotpotqa"],
default="natural_questions")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--data-path", type=str, default="/app/data/ICD-main-1/data")

    args = parser.parse_args()
    amateur_model_name = args.amateur_model_name
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    mode = args.mode
    list_data_dict, labels = load_csv(args.dataset_name, args.debug)
    dataset = list_data_dict
    num_samples = len(dataset)
    sample_ratio = 0.2
    sample_size = int(num_samples * sample_ratio)
    indices = np.random.choice(num_samples, sample_size, replace=False)
    sampled_dataset = [dataset[i] for i in indices]
    sampled_labels = [labels[i] for i in indices]
    

    llm = ContrastiveDecoding(model_name=model_name, amateur_model_name=amateur_model_name, device=device)
    stop_word_list = ["Q:"]
    llm.set_stop_words(stop_word_list)

    result_dict = {'qid_list': [], 'answers': {}, 'model_completion': {}, 'questions': {}, 'logit_scores': {}}

    os.makedirs(args.data_path, exist_ok=True)

    try:
        permute_idx = np.load(os.path.join(args.data_path, "val_test_idx_{}.npy"))
    except:
        permute_idx = np.random.permutation(len(list_data_dict))
        np.save(os.path.join(args.data_path, "val_test_idx_{}.npy"), permute_idx)

    # val_idx = permute_idx[0:100]
    # test_idx = permute_idx[100:]

    # val_idx = permute_idx[0:int(len(list_data_dict)*.2)]
    # test_idx = permute_idx[int(len(list_data_dict)*.2):]

    # val_dataset = [list_data_dict[i] for i in val_idx]
    # test_dataset = [list_data_dict[idx] for idx in test_idx]

    # val_label = [labels[i] for i in val_idx]
    # test_label = [labels[idx] for idx in test_idx]
    # dataset=list_data_dict
    # if args.val_test_mode=='val':
    #     dataset=val_dataset
    #     labels=val_label
    # elif args.val_test_mode=='test':
    #     dataset=test_dataset
    #     labels=test_label

    dataset = list_data_dict
    # dataset=dataset[:10]
    # labels=labels[:10]

    start = time.time()
    for i, question in enumerate(tqdm(sampled_dataset)):
        # for i, question in enumerate(tqdm(val_dataset, desc='Processing')):

        answer = sampled_labels[i]
        prompt=build_prompt(question,args.prompt_style)
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p,top_k=args.top_k, temperature=args.temperature,repetition_penalty=args.repetition_penalty, mode=mode,relative_top=args.relative_top)
        model_completion, c_dist = llm.generate(prompt, **generate_kwargs)
        # pdb.set_trace()
        logit_scores = 0
        # if mode=='baseline' or mode=='dola' or mode=='with_dola':
        #     logit_scores=0
        # else:
        #     logit_scores = llm.get_lm_scores_from_outputs(outputs, mode=mode)

        # process output format to remove unnecessary tokens; designed for few-shot prompt
        for stop_word in stop_word_list:
            length_to_remove = len(stop_word)
            if model_completion[-length_to_remove:] == stop_word:
                model_completion = model_completion[:-length_to_remove]
        if 'Q:' in model_completion:
            model_completion = model_completion.split('Q:')[0].strip()
        model_completion = model_completion.strip()


        result_dict['qid_list'].append(i)
        result_dict['answers'][i] = answer
        result_dict['model_completion'][i] = model_completion
        result_dict['questions'][i] = question
        result_dict['logit_scores'][i] = logit_scores

        if args.debug:
            if i > 10:
                break

                # here I note the next 'print' lines
    '''
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')



        print(f'Question: {sample}\n\n'
            f'Model Completion: {model_completion}\n\n')

        print(f'Num of total question: {len(answers)}.')
    if mode == "dola" or mode=="activation" and args.debug:
        total_tokens = sum(premature_layer_dist.values())
        if total_tokens > 0:
            for l in candidate_premature_layers:

                print('Premature layer {0} was used {1} times, {2}%'.format(l, premature_layer_dist[l], round(premature_layer_dist[l] / total_tokens * 100, 2)))
    '''

    end=time.time()
    elapsed_time = end-start
    print(f"time:{end-start}s")

    # pdb.set_trace()
    # save results to a json file
    # model_tag = "llama-7b" from model_name "huggyllama/llama-7b"
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path
    with open('runtime_ihcd.txt', 'a') as f:
        f.write(f"time:{elapsed_time:.2f}seconds\n")
