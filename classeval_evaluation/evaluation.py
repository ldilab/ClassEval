import argparse
import json
import os
from pathlib import Path

from path_util import PathUtil
from test_pipeline import AutoTest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id_key",
        type=str,
        default="task_id",
        help="id key",
    )
    parser.add_argument(
        "--pred_key",
        type=str,
        default="code",
        help="prediction key",
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        help="root directory of the project",
    )
    parser.add_argument(
        "--source_file_name",
        type=str,
        default="model_output",
        help="source of model output",
    )
    parser.add_argument(
        "--greedy",
        type=int,
        default=1,
        help="whether the model result is greedy or not",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default="ClassEval_data",
        help="ClassEval data",
    )
    args = parser.parse_args()

    AutoT = AutoTest(args.eval_data, args.root_dir, args.id_key, args.pred_key)
    source_file_name = Path(args.source_file_name)
    model_name = source_file_name.stem
    model_list = [model_name]
    AutoT.test_pipeline(model_name, source_file_name)

    AutoT.evaluate(model_list)
    result = {}
    if args.greedy == 1:
        result["pass_1_greedy"] = AutoT.cal_metrics_pass_at_k(model_list, 1, 1)
    else:
        result["pass_1"] = AutoT.cal_metrics_pass_at_k(model_list, 1, 5)
        result["pass_3"] = AutoT.cal_metrics_pass_at_k(model_list, 3, 5)
        result["pass_5"] = AutoT.cal_metrics_pass_at_k(model_list, 5, 5)
    save_path = source_file_name.parent / f"{model_name}_eval.json"

    with open(save_path, "w") as f:
        json.dump(result, f, indent=4, sort_keys=True)
