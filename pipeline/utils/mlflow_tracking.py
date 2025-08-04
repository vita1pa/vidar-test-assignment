from mlflow.tracking import MlflowClient
import yaml
import json


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)
    
def init_mlflow():
    pass

def get_mlflow_client():
    params = load_params()
    tracking_uri = params["mlflow"]["tracking_uri"]
    return MlflowClient(tracking_uri=tracking_uri)

def get_experiment_id(client: MlflowClient):
    params = load_params()
    experiment_name = params["mlflow"]["experiment_name"]
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return client.create_experiment(experiment_name)
    return experiment.experiment_id

def create_and_save_parent_run(client: MlflowClient):
    experiment_id = get_experiment_id(client)
    run = client.create_run(experiment_id)
    run_id = run.info.run_id
    # Сохраняем parent run_id в файл
    with open("parent_run_id.json", "w") as f:
        json.dump({"parent_run_id": run_id}, f)
    return run_id

def load_parent_run_id():
    with open("parent_run_id.json", "r") as f:
        return json.load(f)["parent_run_id"]

def create_child_run(client: MlflowClient, 
                     parent_run_id: str, stage_name: str):
    experiment_id = get_experiment_id(client)
    run = client.create_run(experiment_id, tags={"mlflow.parentRunId": parent_run_id, "stage": stage_name})
    run_id = run.info.run_id
    return run_id

def terminate_run(client: MlflowClient, run_id: str, status="FINISHED"):
    client.set_terminated(run_id, status)


def main():

    # Подключаемся к клиенту
    client = get_mlflow_client()

    # Создаем и сохраняем parent run
    parent_run_id = create_and_save_parent_run(client)
    print(f"Created parent run with ID: {parent_run_id}")

if __name__ == "__main__":
    main()