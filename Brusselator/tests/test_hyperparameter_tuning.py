from types import SimpleNamespace

from src import config
from src import hyperparameter_tuning as tuning


def test_describe():
    """Run names must follow the key=value convention without spaces."""
    changes = {
        "hidden_layer_sizes": (128, 128, 128),
        "activation": "tanh",
        "learning_rate_init": 0.001,
    }

    result = tuning.describe(changes)

    expected = (
        "hidden_layer_sizes=(128,128,128)_"
        "activation=tanh_"
        "learning_rate_init=0.001"
    )

    assert result == expected


def test_tune_logic(monkeypatch):
    """
    Test the tuning workflow using fake training and logging functions.

    Check that:
    - Duplicate configurations are skipped.
    - The best validation score is tracked correctly.
    - Models are logged only when a new best score is achieved.
    """
    # Define a small artificial search space.
    monkeypatch.setattr(
        tuning,
        "STAGE_GRIDS",
        {
            # Stage 1: two configurations.
            1: {"learning_rate_init": [0.01, 0.02]},

            # Stage 2: 0.02 is a duplicate of the current best.
            2: {"learning_rate_init": [0.02, 0.03, 0.04]},
        },
    )

    # Prescribed validation scores for each configuration.
    scripted_scores = {
        0.01: 0.50,
        0.02: 0.40,
        0.03: 0.45,
        0.04: 0.35,
    }

    training_calls = []
    logging_calls = []

    def fake_train_model(hyperparameters):
        learning_rate = hyperparameters["learning_rate_init"]

        training_calls.append(dict(hyperparameters))

        return SimpleNamespace(
            hyperparameters=dict(hyperparameters),
            val_score=scripted_scores[learning_rate],
            epochs_run=10,
            training_seconds=1.0,
        )

    monkeypatch.setattr(tuning, "train_model", fake_train_model)


    def fake_log_result(
        result,
        experiment,
        run_name,
        tags,
        log_model,
    ):
        run_id = f"run-{len(logging_calls) + 1}"

        logging_calls.append({
            "run_name": run_name,
            "score": result.val_score,
            "experiment": experiment,
            "tags": tags,
            "log_model": log_model,
            "run_id": run_id,
        })

        return run_id

    monkeypatch.setattr(tuning, "log_result", fake_log_result)

    # Capture the final best configuration instead of printing it.
    final_state = {}

    def fake_print_best(best, improvements):
        final_state["best"] = dict(best)
        final_state["improvements"] = list(improvements)

    monkeypatch.setattr(tuning, "print_best", fake_print_best)


    tuning.tune()
    # Check duplicate configurations are skipped.
    # Stage 1: 0.01, 0.02
    # Stage 2: 0.02 (duplicate), 0.03, 0.04
    #
    # Therefore, only four configurations should be trained.

    assert len(training_calls) == 4

    tested_learning_rates = [
        params["learning_rate_init"]
        for params in training_calls
    ]

    assert tested_learning_rates == [0.01, 0.02, 0.03, 0.04]

    # Check that the best score is tracked correctly.
    best = final_state["best"]

    assert best["score"] == 0.35
    assert best["hyperparameters"]["learning_rate_init"] == 0.04
    assert best["run_name"] == "stage2_learning_rate_init=0.04"
    assert best["run_id"] == "run-4"

    # Only runs that improved the best score belong to the
    # improvement path: 0.50 -> 0.40 -> 0.35.

    improvements = final_state["improvements"]

    assert len(improvements) == 3

    expected_improvement_scores = [0.50, 0.40, 0.35]

    for improvement, expected_score in zip(
        improvements, expected_improvement_scores
    ):
        assert improvement[1] == expected_score

    # Check model logging decisions.

    # 0.50: first run, new best -> save
    # 0.40: improvement       -> save
    # 0.45: worse than 0.40   -> do not save
    # 0.35: improvement       -> save

    assert len(logging_calls) == 4

    assert [
        call["log_model"] for call in logging_calls
    ] == [True, True, False, True]

    # Every configuration must still be logged as an MLflow run.
    assert [
        call["run_name"] for call in logging_calls
    ] == [
        "stage1_learning_rate_init=0.01",
        "stage1_learning_rate_init=0.02",
        "stage2_learning_rate_init=0.03",
        "stage2_learning_rate_init=0.04",
    ]

    # All runs must use the tuning experiment.
    assert all(
        call["experiment"] == config.MLFLOW_TUNING_EXPERIMENT
        for call in logging_calls
    )
