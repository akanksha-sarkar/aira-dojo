from dojo.analysis.llm_as_judge import load_program, judge_program
import os
import glob
import json
def analyze_run(program_path: str, program_id: str):
    program = load_program(program_path)
    result = judge_program(program, program_id=program_id)
    return result

def extract_program_id(program_path: str):
    return os.path.basename(program_path)[12:-3]

def analyze_all_programs(experiment_path: str):
    count = 0
    program_paths = glob.glob(os.path.join(experiment_path, "*.py"))
    analysis_dict = {}
    for i, program_path in enumerate(program_paths):
        count += 1
        print(f"Analyzing program {count} of {len(program_paths)}")
        program_name = os.path.basename(program_path)
        program_id = extract_program_id(program_path)
        # Get test accuracy of program
        eval_name = "eval_step" + program_id + ".json"

        assert os.path.exists(os.path.join(experiment_path, eval_name)), f"Eval file {eval_name} does not exist"
        with open(os.path.join(experiment_path, eval_name), "r") as f:
            eval_result = json.load(f)
        test_accuracy = eval_result["acc"]
        print(f"Test accuracy of program {program_id}: {test_accuracy}")
        # Get LLM analysis of program
        program_analysis_output = analyze_run(program_path, program_id)
        program_analysis = program_analysis_output["parsed_output"]
        prompt_tokens = program_analysis_output["usage"]["prompt_tokens"]
        completion_tokens = program_analysis_output["usage"]["completion_tokens"]
        total_tokens = program_analysis_output["usage"]["total_tokens"]
        print(f"Finished analyzing program {program_id}")
        # Check for model usage
        program_text = load_program(program_path)
        if "timm/vit_base_patch14_reg4_dinov2.lvd142m" in program_text:
            used_dinov2 = True
        else:
            used_dinov2 = False
        if "timm/vit_base_patch16_clip_224.openai" in program_text:
            used_clip = True
        else:
            used_clip = False
        program_analysis["used_dinov2"] = used_dinov2
        program_analysis["used_clip"] = used_clip
        analysis_dict[program_id] = {
                                      "program_name": program_name, 
                                      "eval_name": eval_name,
                                      "test_accuracy": test_accuracy,
                                      "program_analysis": program_analysis,
                                      "prompt_tokens": prompt_tokens,
                                      "completion_tokens": completion_tokens,
                                      "total_tokens": total_tokens,
                                    }
    analysis_dict["total_programs"] = count
    analysis_dict["experiment_path"] = experiment_path
    return analysis_dict

def total_cost(prompt_tokens, completion_tokens,
               prompt_price_per_1m, completion_price_per_1m):
    """
    Returns total cost in dollars.

    Example prices:
        prompt_price_per_1m=5.00
        completion_price_per_1m=15.00
    """
    return (
        (prompt_tokens / 1_000_000) * prompt_price_per_1m
        + (completion_tokens / 1_000_000) * completion_price_per_1m
    )

def total_experiment_cost(results_dict,
                          prompt_price_per_1m,
                          completion_price_per_1m):
    total = 0.0
    for run in results_dict.values():
        if type(run) != dict:
            continue
        total += total_cost(
            run.get("prompt_tokens", 0),
            run.get("completion_tokens", 0),
            prompt_price_per_1m,
            completion_price_per_1m,
        )
        print(f"Total cost of run {run['program_name']}: ${total_cost(run['prompt_tokens'], run['completion_tokens'], prompt_price_per_1m, completion_price_per_1m)}")
    return total

if __name__ == "__main__":
    dataset = "resisc45"
    setting = "aSSL_backbone_unsupervised"
    k = 2
    seed = 0
    run_number = 2
    all_test_programs_path = f"/share/j_sun/as2637/logs/{dataset}/{setting}/k{k}/seed{seed}/aira/{run_number}/all_test_programs"
    program_path = f"/share/j_sun/as2637/logs/{dataset}/{setting}/k{k}/seed{seed}/aira/{run_number}/all_test_programs/program_step0004_7694c9d8.py"
    save_path = f"src/dojo/analysis/runs/{dataset}/{setting}/k{k}/seed{seed}/aira/{run_number}"
    if os.path.exists(save_path):
        print(f"Analysis already exists at {save_path}")
    else:
        os.makedirs(save_path, exist_ok=True)
        analysis_dict = analyze_all_programs(all_test_programs_path)
        with open(os.path.join(save_path, f"analysis_dict.json"), "w") as f:
            json.dump(analysis_dict, f, indent=4)
    print("Computing total cost of experiment...")
    with open(f"{save_path}/analysis_dict.json", "r") as f:
        analysis_dict = json.load(f)
    total_cost = total_experiment_cost(analysis_dict,
                                       prompt_price_per_1m=1.10,
                                       completion_price_per_1m=4.40)
    print(f"Total cost of experiment: ${total_cost}")