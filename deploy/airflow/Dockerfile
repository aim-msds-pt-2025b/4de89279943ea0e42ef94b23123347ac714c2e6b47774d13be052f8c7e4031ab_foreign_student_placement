FROM apache/airflow:2.9.3
COPY requirements.txt /
USER root
# install only the DockerOperator plugin
RUN pip install --no-cache-dir apache-airflow-providers-docker

USER airflow
