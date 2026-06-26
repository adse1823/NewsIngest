from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "finplatform",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="fin_pipeline",
    default_args=default_args,
    description="Daily feature engineering, NLP, GNN, and drift monitoring",
    schedule_interval="0 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["finplatform"],
) as dag:

    feature_export = BashOperator(
        task_id="feature_export",
        bash_command="cd /opt/airflow && python feature_store/export.py",
    )

    nlp_sentiment = BashOperator(
        task_id="nlp_sentiment",
        bash_command="cd /opt/airflow && python nlp/sentiment.py",
    )

    nlp_embeddings = BashOperator(
        task_id="nlp_embeddings",
        bash_command="cd /opt/airflow && python nlp/embeddings.py",
    )

    build_graph = BashOperator(
        task_id="build_graph",
        bash_command="cd /opt/airflow && python graph/build_graph.py",
    )

    train_gnn = BashOperator(
        task_id="train_gnn",
        bash_command="cd /opt/airflow && python graph/train_gnn.py",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/airflow && python modeling/train.py",
    )

    drift_report = BashOperator(
        task_id="drift_report",
        bash_command="cd /opt/airflow && python monitoring/drift_report.py",
    )

    feature_export >> nlp_sentiment >> nlp_embeddings >> build_graph >> train_gnn >> train_model >> drift_report
