import mlflow
from pathlib import Path

mlflow.set_tracking_uri(Path("./mlruns").resolve().as_uri())
mlflow.MlflowClient().set_registered_model_alias("fin-platform-lgbm", "champion", "8")
print("champion alias set on version 8")
