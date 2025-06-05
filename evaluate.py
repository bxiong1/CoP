from judge import harmbench_judge
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
data_df = pd.read_csv("/workspace/rl_llm_new/harmbench_behaviors_sampled_queries.csv")
id = data_df.BehaviorID
prompt_l = data_df.Behavior.tolist()
cls = AutoModelForCausalLM.from_pretrained("/workspace/HarmBench-Llama-2-13b-cls", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("/workspace/HarmBench-Llama-2-13b-cls", use_fast=False, truncation_side="left", padding_side="left")

jb_count = 0
problem_id = []
for i in range(len(id)):
    try:
        results = pd.read_csv(f"/workspace/o1_test_score_10_harmbench_saved_all_grok/o1_jb_{id[i]}.csv")
        behaviors = results["best_msg"]
        responses = results["jailbreak_output"]
        goal=prompt_l[i]
        judge_l = []
        for j in range(len(behaviors)):
            judge = harmbench_judge(cls, tokenizer, behaviors[j], responses[j])[0]
            judge_l.append(judge)
        if any("Yes" in item for item in judge_l):
            # print("Questions")
            # print(behavior)
            # print("\n")
            # print("Response")
            # print(response)
            # print("\n")
            # print("ID")
            # print(id[i])
            jb_count+=1
            
        else:
            problem_id.append(id[i])
            continue
    except:
        continue
print(jb_count)
print(jb_count/400)