import argparse
import itertools

from config import dataset_paths, models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", 
        type=str, 
        nargs='+',
        required=True,
        help="Dataset name(s) (use 'all' for all datasets)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        nargs='+',
        required=True,
        help="Model name(s) (use 'all' for all models)"
    )
    parser.add_argument("--debug", action='store_true')
    
    args = parser.parse_args()

    # Resolve datasets
    if 'all' in args.dataset:
        selected_datasets = list(dataset_paths.keys())
    else:
        selected_datasets = args.dataset
        # Validate
        for d in selected_datasets:
            if d not in dataset_paths:
                raise ValueError(f"Unknown dataset: {d}. Available: {list(dataset_paths.keys())}")

    # Resolve models
    if 'all' in args.model:
        selected_models = list(models.keys())
    else:
        selected_models = args.model
        for m in selected_models:
            if m not in models:
                raise ValueError(f"Unknown model: {m}. Available: {list(models.keys())}")

    print(f"Running evaluation for models: {selected_models}")
    print(f"on datasets: {selected_datasets}")

    # Evaluate all combinations
    for model_name, dataset_name in itertools.product(selected_models, selected_datasets):
        print(f"\n{'='*60}")
        print(f"Evaluating model '{model_name}' on dataset '{dataset_name}'")
        print(f"{'='*60}")

        dataset_path = dataset_paths[dataset_name]
        ModelClass = models[model_name]
        model = ModelClass()

        try:
            output_csv = model.inference_tsv(dataset_path, debug_mode=args.debug)
            model.eval_results(output_csv, dataset_name, debug_mode=args.debug)
        except Exception as e:
            print(f"Failed on Combination {model_name} / {dataset_name}: {e}")
            continue


if __name__ == "__main__":
    main()