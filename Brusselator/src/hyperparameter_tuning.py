"""
Hyperparameter search in stages (greedy): each stage varies only its own hyperparameters,
starting from the best configuration found so far.

Every combination is one MLflow run in the tuning experiment. A model is saved only when
a run becomes the new best (the "improvement path"). A configuration that has already
been run in this session is skipped.

Run:  python -m src.hyperparameter_tuning
"""

import json
import time

from sklearn.model_selection import ParameterGrid

from . import config
from .train_neural_network import DEFAULT_HYPERPARAMETERS, log_result, train_model


# Each stage: hyperparameter name -> values (crossed with ParameterGrid).
# Stages run in order; to enable a stage, uncomment it.
STAGE_GRIDS = {
    # Stage 1: capacity (width x depth), 9 runs
    1: {"hidden_layer_sizes": [(width,) * depth for depth in (3, 4, 5) for width in (128, 256, 512)]},
    # Stage 2: optimisation, 3 x 3 = 9 combinations (the one equal to the best so far is skipped)
    # 2: {"learning_rate_init": [3e-4, 1e-3, 3e-3], "batch_size": [64, 128, 256]},
    # Stage 3: activation and regularisation, 2 x 4 = 8 combinations
    # 3: {"activation": ["tanh", "relu"], "alpha": [1e-5, 1e-4, 1e-3, 1e-2]},
}


def describe(changes):
    return "_".join(f"{key}={str(value).replace(' ', '')}" for key, value in changes.items())


def print_best(best, improvements):
    print("\n=== Best so far")
    if best["run_id"] is None:
        print("No run finished.")
        return
    print(f"Configuration: {best['hyperparameters']}")
    print(f"Validation mean relative L2 error: {best['score']:.4e}")
    print(f"Run: {best['run_name']} | run id: {best['run_id']}")
    print("Improvement path (models saved):")
    for run_name, score, run_id in improvements:
        print(f"  {score:.4e} | {run_name} | {run_id}")
    print(f"Evaluate with:  python -m src.evaluate --run-id {best['run_id']}")


def tune():
    session = time.strftime("%Y%m%d-%H%M%S")

    best = {"score": float("inf"), "hyperparameters": dict(DEFAULT_HYPERPARAMETERS),
            "run_id": None, "run_name": None}
    results = {}        # configuration (as JSON text) -> validation score, for runs done in this session
    improvements = []   # (run_name, score, run_id) of every run that became the new best

    try:
        for stage, grid in STAGE_GRIDS.items():
            base = dict(best["hyperparameters"])
            combinations = list(ParameterGrid(grid))
            print(f"\n=== Stage {stage}: {len(combinations)} combinations")
            print(f"Base configuration: {base}")

            for i, changes in enumerate(combinations, start=1):
                hyperparameters = {**base, **changes}
                key = json.dumps(hyperparameters, sort_keys=True)
                run_name = f"stage{stage}_{describe(changes)}"

                if key in results:
                    print(f"\n[{i}/{len(combinations)}] {run_name}: skipped "
                          f"(already run, validation mean relative L2 error {results[key]:.4e})")
                    continue

                print(f"\n[{i}/{len(combinations)}] {run_name}")
                result = train_model(hyperparameters)
                score = result.val_score
                is_best = score < best["score"]
                run_id = log_result(
                    result,
                    experiment=config.MLFLOW_TUNING_EXPERIMENT,
                    run_name=run_name,
                    tags={"stage": str(stage), "tuning_session": session},
                    log_model=is_best,      # save the model only if it beats the best so far
                )
                results[key] = score

                message = (f"  validation mean relative L2 error: {score:.4e} "
                           f"({result.epochs_run} epochs, {result.training_seconds / 60:.1f} min)")
                if is_best:
                    best = {"score": score, "hyperparameters": result.hyperparameters,
                            "run_id": run_id, "run_name": run_name}
                    improvements.append((run_name, score, run_id))
                    message += "   <- new best (model saved)"
                print(message)
    finally:
        # Also reached on Ctrl+C or an error: the best run so far is always reported
        print_best(best, improvements)


def main():
    start = time.time()
    tune()
    print(f"\nTotal computation time: {(time.time() - start) / 60:.2f} minutes.")


if __name__ == "__main__":
    main()