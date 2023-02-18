import mlflow

mlflow.set_tracking_uri("http://mlflow.hdfcbdexgmemcsht.westeurope.azurecontainer.io:80")

with mlflow.start_run() as run:
    mlflow.log_metric("t", 1)
    mlflow.log_artifact("setup.py")
