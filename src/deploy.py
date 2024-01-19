import os 

import boto3
import sagemaker
import pandas as pd
from sagemaker.sklearn import SKLearn

AWS_ACCOUNT = os.environ["AWS_ACCOUNT"]
sagemaker_role = f"arn:aws:iam::{AWS_ACCOUNT}:role/ml-sagemaker-execution-role"

def get_all_params():
    boto_session = boto3.session.Session(
        region_name='us-east-2', 
        )
    sess = sagemaker.Session(boto_session)
    bucket = sess.default_bucket()                  
    PREFIX = 'sklearn-boston-housing'

    training = sess.upload_data(path='./data/input/housing.csv', key_prefix=PREFIX + "/training")
    output = 's3://{}/{}/output/'.format(bucket,PREFIX)
    train_params = {
        "sagemaker_session": sess,
        "training_path": training,
        "output_path": output
    }
    return train_params


def train(train_params):
    sk = SKLearn(
        entry_point='sklearn-boston-housing.py',
        source_dir="./src/train_serve/",
        role=sagemaker_role,
        framework_version='0.20.0',
        instance_count=1, 
        instance_type='ml.m5.large',
        output_path=train_params["output_path"],
        hyperparameters={
            'normalize': True,
            'test-size': 0.1
        }
    )

    sk.fit({'training': train_params["training_path"]})
    return sk


def predict_on_endpoint(sk):
    try:
        sk_predictor = sk.deploy(initial_instance_count=1, instance_type='ml.t2.medium')

        data = pd.read_csv('./data/input/housing.csv')
        payload = data[:5].drop(['medv'], axis=1) 
        payload = payload.to_csv(header=False, index=False)

        sk_predictor.serializer = sagemaker.serializers.CSVSerializer()
        sk_predictor.deserializer = sagemaker.deserializers.CSVDeserializer()

        response = sk_predictor.predict(payload)
        print(response)
    finally:
        if sk_predictor:
            sk_predictor.delete_endpoint()


if __name__ == "__main__":
    params = get_all_params()
    model_params = train(params)
    predict_on_endpoint(model_params)