import json
import os
import pandas as pd
def get_eval_sum(output_dir):
    print(f"Processing {output_dir}...")
    results_base = json.load(open(os.path.join(output_dir, "results_base.json"), "r"))
    results_final = json.load(open(os.path.join(output_dir, "results_final.json"), "r"))
    return results_base, results_final

if __name__ == "__main__":
    output_dir = "/share/home/sxjiang/myproject/self-learn/eval/results/2_6_bx"
    sum_results = []
    for experiment_name in os.listdir(output_dir):
        if os.path.isfile(os.path.join(output_dir, experiment_name)):
            continue
        if "results_final.json" not in os.listdir(os.path.join(output_dir, experiment_name)):
            continue
        _, results_final = get_eval_sum(os.path.join(output_dir, experiment_name))
        new_result_item = {"experiment_name": experiment_name}
        
        for cat in ["Easy", "Medium", "Hard"]:
            new_result_item[f"{cat}_peak_acc"] = results_final[cat]["peak_acc"]
            new_result_item[f"{cat}_adapt_acc"] = results_final[cat]["adapt_acc"]
            new_result_item[f"{cat}_avg_steps"] = results_final[cat]["avg_steps"]
        sum_results.append(new_result_item)
    pd.DataFrame(sum_results).to_csv(os.path.join(output_dir, "sum_results.csv"), index=False)