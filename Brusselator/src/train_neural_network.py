"""
Train one model on the training data (train_model) and log it to MLflow (log_result).
Training and logging are separate, so the caller decides what is logged.

Run:  python -m src.train_neural_network
"""

import time
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .dataset import dataset_path, load_dataset, load_metadata
from .metrics import flatten_summary, print_summary, summarize_errors


DEFAULT_HYPERPARAMETERS = dict(
    hidden_layer_sizes=(128, 128, 128),
    activation="tanh",
    solver="adam",
    learning_rate_init=1e-3,
    batch_size=128,
    alpha=1e-4,               # L2 regularisation (sklearn default, listed so that it is explicit)
    max_iter=500,
    n_iter_no_change=20,      # early stopping patience (epochs)
    tol=1e-6,                 # minimum improvement of the internal validation score
)


@dataclass
class TrainingResult:
    """Everything produced by one training run."""
    model: object
    hyperparameters: dict
    train_summary: dict       # relative L2 error on the training set (all samples and per regime)
    val_summary: dict         # relative L2 error on the validation set
    loss_curve: list          # training loss per epoch (scaled space)
    validation_scores: list   # internal early-stopping validation score per epoch (R^2)
    epochs_run: int
    n_train: int
    n_validation: int
    training_seconds: float

    @property
    def val_score(self):
        """Mean relative L2 error on the validation set (lower is better)."""
        return self.val_summary["all"]["mean"]


def build_model(hyperparameters=None, seed=config.SEED):
    """
    Input scaling and output scaling are both handled here.

    SCALING:
    - Inputs  (A, B, x0, y0): StandardScaler inside the Pipeline.
    - Targets (x(t) and y(t) columns): StandardScaler via TransformedTargetRegressor.
      It scales the targets before training and applies the inverse transform in
      predict(), so model.predict(...) returns values in the ORIGINAL units (directly
      comparable with Runge-Kutta). The MSE training loss is computed on scaled targets.
    """
    params = {**DEFAULT_HYPERPARAMETERS, **(hyperparameters or {})}
    mlp = MLPRegressor(
        **params,
        early_stopping=True,            # holds out validation_fraction of the TRAINING data
        validation_fraction=0.1,
        random_state=seed,
    )
    regressor = Pipeline([
        ("input_scaler", StandardScaler()),
        ("mlp", mlp),
    ])
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),   # <-- target scaling
    )


def train_model(hyperparameters=None):
    """Train one model and compute its training / validation errors."""
    params = {**DEFAULT_HYPERPARAMETERS, **(hyperparameters or {})}

    train_parameters, train_ODE_data, _ = load_dataset(dataset_path("train"))
    val_parameters, val_ODE_data, _ = load_dataset(dataset_path("validation"))

    print("Training model ...")
    
    model = build_model(params)
    start = time.perf_counter()
    model.fit(train_parameters, train_ODE_data)
    training_seconds = time.perf_counter() - start
    mlp = model.regressor_.named_steps["mlp"]

    # Predictions are in original units (scaling is undone by the model)
    train_summary = summarize_errors(train_parameters, train_ODE_data,
                                     model.predict(train_parameters))
    val_summary = summarize_errors(val_parameters, val_ODE_data,
                                   model.predict(val_parameters))

    return TrainingResult(
        model=model,
        hyperparameters=params,
        train_summary=train_summary,
        val_summary=val_summary,
        loss_curve=list(mlp.loss_curve_),
        validation_scores=list(mlp.validation_scores_),
        epochs_run=int(mlp.n_iter_),
        n_train=len(train_parameters),
        n_validation=len(val_parameters),
        training_seconds=training_seconds,
    )


def log_result(result, experiment=config.MLFLOW_EXPERIMENT, run_name=None, tags=None,
               log_model=True):
    """Log a TrainingResult to MLflow (params, metrics, curves and, if wanted, the model).
    Returns the MLflow run id."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(result.hyperparameters)
        mlflow.log_param("seed", config.SEED)
        if tags:
            mlflow.set_tags(tags)

        # Settings of the dataset the model was trained on (read from the data file)
        mlflow.log_params({f"data_{key}": value
                           for key, value in load_metadata(dataset_path("train")).items()})
        mlflow.log_params({"n_train": result.n_train, "n_validation": result.n_validation})

        mlflow.log_metrics({
            **flatten_summary("train", result.train_summary),
            **flatten_summary("validation", result.val_summary),
            "epochs_run": result.epochs_run,
            "training_seconds": result.training_seconds,
        })
        for epoch, (loss, score) in enumerate(zip(result.loss_curve, result.validation_scores)):
            mlflow.log_metrics({"training_loss": loss, "internal_validation_score": score},
                               step=epoch)

        if log_model:
            mlflow.sklearn.log_model(result.model, name="model",
                                     skops_trusted_types=config.SKOPS_TRUSTED_TYPES)
            mlflow.set_tag("model_logged", "true")

    return run.info.run_id


def main():
    start = time.time()
    result = train_model()
    run_id = log_result(result)

    print_summary("Validation set", result.val_summary)
    print(f"\nStopped after {result.epochs_run} epochs ({result.training_seconds / 60:.2f} minutes of training)")
    print(f"MLflow run id: {run_id}")
    print(f"Total computation time: {(time.time() - start) / 60:.2f} minutes.")


if __name__ == "__main__":
    main()