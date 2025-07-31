import mlflow
import yaml


def init_mlflow(config_path="config.yaml"):
    """Initialize MLflow with settings from config.yaml."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])