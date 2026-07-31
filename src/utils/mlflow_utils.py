import mlflow
import mlflow.sklearn
from src.utils.mlflow_utils import setup_mlflow


def setup_mlflow():

    mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment("Customer Segmentation")


setup_mlflow()