"""Train one model on the training data and log it, with its metrics, to MLflow."""

import time

import mlflow
import mlflow.sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .dataset import dataset_path, load_metadata, load_dataset
from .metrics import flatten_summary, print_summary, summarize_errors


DEFAULT_HYPERPARAMETERS = dict(
    hidden_layer_sizes=(128, 128, 128),
    activation="tanh",
    solver="adam",
    learning_rate_init=1e-3,
    batch_size=128,
    max_iter=500,
    n_iter_no_change=20,      # early stopping patience (epochs)
    tol=1e-6,                 # minimum improvement of the internal validation score
)


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


def train_model(hyperparameters=None, run_name=None):
    """Train one model, log params / metrics / model to MLflow.
    Returns the fitted model, the validation error summary and the MLflow run id."""
    params = {**DEFAULT_HYPERPARAMETERS, **(hyperparameters or {})}

    train_parameters, train_ODE_data, _ = load_dataset(dataset_path("train"))
    val_parameters, val_ODE_data, _ = load_dataset(dataset_path("validation"))

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_param("seed", config.SEED)
        # Settings of the dataset the model is trained on (read from the data file)
        mlflow.log_params({f"data_{key}": value
                           for key, value in load_metadata(dataset_path("train")).items()})
        mlflow.log_params({"n_train": len(train_parameters), "n_validation": len(val_parameters)})

        print("Training model ...\n")

        model = build_model(params)
        model.fit(train_parameters, train_ODE_data)
        mlp = model.regressor_.named_steps["mlp"]

        

        # Predictions are in original units (scaling is undone by the model)
        train_summary = summarize_errors(train_parameters, train_ODE_data,
                                         model.predict(train_parameters))
        val_summary = summarize_errors(val_parameters, val_ODE_data,
                                       model.predict(val_parameters))

        mlflow.log_metrics({
            **flatten_summary("train", train_summary),
            **flatten_summary("validation", val_summary),
            "epochs_run": mlp.n_iter_,
        })
        for epoch, (loss, score) in enumerate(zip(mlp.loss_curve_, mlp.validation_scores_)):
            mlflow.log_metrics({"training_loss": loss, "internal_validation_score": score},
                               step=epoch)

        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=config.SKOPS_TRUSTED_TYPES)

    return model, val_summary, run.info.run_id


def main():
    start = time.time()
    _, val_summary, run_id = train_model()
    print_summary("Validation set", val_summary)
    print(f"\nMLflow run id: {run_id}")
    print(f"Total computation time: {(time.time() - start) / 60:.2f} minutes.")


if __name__ == "__main__":
    main()