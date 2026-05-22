import json
import argparse
from copy import deepcopy


BASE_INSTANCE_KEYS = {"input", "output"}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_prediction_keys(instance):
    """
    Return all non-base keys from an instance.
    """
    return set(instance.keys()) - BASE_INSTANCE_KEYS


def merge_json_files(base_data, new_data):
    """
    Merge prediction keys from new_data into base_data.
    """

    # Lookup tasks by task_name
    new_lookup = {
        task["task_name"]: task
        for task in new_data
    }

    merged = deepcopy(base_data)

    for task in merged:
        task_name = task["task_name"]

        if task_name not in new_lookup:
            print(f"Skipping missing task: {task_name}")
            continue

        new_task = new_lookup[task_name]

        base_instances = task["instances"]
        new_instances = new_task["instances"]

        if len(base_instances) != len(new_instances):
            raise ValueError(
                f"Different number of instances for task {task_name}"
            )

        for i, (base_inst, new_inst) in enumerate(
            zip(base_instances, new_instances)
        ):

            # Optional sanity checks
            if base_inst.get("input") != new_inst.get("input"):
                raise ValueError(
                    f"Input mismatch in task {task_name}, instance {i}"
                )

            # Copy over all non-base keys
            pred_keys = get_prediction_keys(new_inst)

            for key in pred_keys:
                if key in base_inst:
                    print(
                        f"Warning: overwriting key '{key}' "
                        f"in task {task_name}, instance {i}"
                    )

                base_inst[key] = new_inst[key]

    return merged


def run(args_list):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_json",
        required=True,
        help="Existing merged/base JSON file"
    )

    parser.add_argument(
        "--new_json",
        required=True,
        help="JSON file containing additional keys to merge"
    )

    parser.add_argument(
        "--output_json",
        required=True
    )

    args, _ = parser.parse_known_args(args_list)

    base_data = load_json(args.base_json)
    new_data = load_json(args.new_json)

    merged = merge_json_files(base_data, new_data)

    with open(args.output_json, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Saved merged file to: {args.output_json}")