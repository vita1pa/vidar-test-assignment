import mlflow


def init_mlflow(tracking_uri: str="sqlite:///mlruns/mlruns.db", 
                experiment_name: str="audrok_pipeline"):
    """Initialize MLflow with settings from config.yaml."""    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)