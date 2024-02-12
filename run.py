import os

from tqdm import tqdm
import numpy as np
import pickle
import random
import time
from functools import partial
from pprint import pprint
import argparse
import copy
from openai import OpenAI
from collections import defaultdict

import torch
import transformers

from utils import *
from hundred_system_prompts import *

index_list = [0, 0, 0, 0, 0]
personas = [_[__] for _, __ in zip([pattern_system_prompts, multiple_choice_system_prompts, persona_system_prompts, memorization_system_prompts, language_system_prompts], index_list)]
other_personas = [_[__:] for _, __ in zip([pattern_system_prompts, multiple_choice_system_prompts, persona_system_prompts, memorization_system_prompts, language_system_prompts], [1, 1, 1, 1, 1])]
for _ in other_personas:
    personas.extend(_)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B')
    parser.add_argument('--agent', type=int, default=-1, choices=[-1, ] + list(range(len(personas))))
    parser.add_argument('--user', type=int, default=-1, choices=[-1, ] + list(range(len(personas))))
    parser.add_argument('--topic', type=int, default=-1, choices=range(len(topics)))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--turns', type=int, default=16)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.agent == -1:
        args.agent = random.randint(0, len(personas)-1)
    if args.user == -1:
        args.user = random.randint(0, len(personas)-1)
    persona, probe_str, judge_func = personas[args.agent]
    user, probe_str_user, judge_func_user = personas[args.user]
    if args.topic == -1:
        args.topic = random.randint(0, len(topics)-1)
    topic = topics[args.topic]
    print(f"Now {args.model_name} chatting over {topic} with system prompts: (A) {persona} and (B) {user}")

    # load assistant
    use_api = "gpt" in args.model_name
    if use_api:
        client = OpenAI()
    else:
        model = ENGINE_MAP[args.model_name]
        tokenizer, intervened_model = load_model(model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=intervened_model,
            tokenizer=tokenizer,
        )
        pipeline.tokenizer.encode = partial(pipeline.tokenizer.encode, add_special_tokens=False)
        
    # task management
    file_name = f"{args.model_name}_agent_{args.agent}_user_{args.user}_turn_{args.turns}"
    file_name += ".pkl"
    
    try:  # resuming halfway jobs if possible
        with open(f"selfchat/{file_name}", "rb") as handle:
            old_pkl = pickle.load(handle)
        pkl = {
            "topic": topic, 
            "history": old_pkl["history"], 
            "probed_history_per_turn": old_pkl["probed_history_per_turn"],
            "seed": args.seed, 
            "persona": persona, 
            "user": user,
        }
    except:
        pkl = {
            "topic": topic, 
            "history": [topic], 
            "probed_history_per_turn": defaultdict(list),
            "seed": args.seed, 
            "persona": persona, 
            "user": user,
        }
    
    for turn in range(len(pkl["history"])+1, args.turns+1):
        pkl_copy = copy.deepcopy(pkl)
        tick = time.time()
        messages = pkl2dict(pkl_copy)
        prompt = llama_v2_prompt(messages)
        print("@"*100)
        print(f"Prompting for the {turn}-th (one-based) turn with prompt:\n{prompt}")
        if use_api:
            completion = client.chat.completions.create(model=args.model_name, messages=messages)
            sequence = completion.choices[0].message.content
        else:
            sequences = pipeline(
                prompt, 
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                num_return_sequences=1,
                eos_token_id=pipeline.tokenizer.eos_token_id,
                max_new_tokens=400,
                return_full_text=False,
                clean_up_tokenization_spaces=True,
            )
            sequence = sequences[0]['generated_text']
        pkl["history"].append(process_answer(sequence))
        tok = time.time()
        print(f"Time taken for turn {turn}: {tok-tick:.2f} seconds")
        if len(pkl["history"]) % 2 == 0:
            with open(f"selfchat/{file_name}", "wb") as handle:
                pickle.dump(pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for turn in range(2, args.turns+1, 2):  # for 2, 4, 6, 8, 10, ...
        runs_to_run = args.runs - len(pkl["probed_history_per_turn"][turn])
        for _ in range(runs_to_run):
            temp_pkl = copy.deepcopy(pkl)
            temp_pkl["history"] = temp_pkl["history"][:turn]
            temp_pkl["history"].append(probe_str)
            pkl_copy = copy.deepcopy(temp_pkl)
            tick = time.time()
            messages = pkl2dict(pkl_copy)
            prompt = llama_v2_prompt(messages)
            if use_api:
                completion = client.chat.completions.create(model=args.model_name, messages=messages)
                sequence = completion.choices[0].message.content
            else:
                sequences = pipeline(
                    prompt, 
                    do_sample=True,
                    top_p=0.9,
                    temperature=1.0,
                    num_return_sequences=1,
                    eos_token_id=pipeline.tokenizer.eos_token_id,
                    max_new_tokens=400,
                    return_full_text=False,
                    clean_up_tokenization_spaces=True,
                )
                sequence = sequences[0]['generated_text']
            pkl["probed_history_per_turn"][turn].append(process_answer(sequence))
            tok = time.time()
            print(f"Time taken for probe turn {turn} ({_+1}/{runs_to_run}): {tok-tick:.2f} seconds")

        with open(f"selfchat/{file_name}", "wb") as handle:
            pickle.dump(pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    pprint(f"Saved to selfchat/{file_name}")

if __name__ == '__main__':
    main()
