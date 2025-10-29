import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import os
import numpy as np
import math
# import holidays
import warnings
import argparse
import sagemaker
import boto3
import re
import s3fs
from sagemaker import AutoML
from sagemaker import get_execution_role
from datetime import datetime, timedelta
from urllib.parse import urlparse
from io import StringIO



def list_csv_files(bucket_name, key_path):
    # List objects within the specified bucket and prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=key_path)
    # Filter out the CSV files
    csv_files = [content['Key'] for content in response.get('Contents', []) if content['Key'].endswith('.csv')]
    return csv_files

def read_csv_files_to_dataframes(bucket_name, csv_files):
    dataframes = []
    for key in csv_files:
        # Get the object from S3
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        # Read the CSV file content
        data = obj['Body'].read().decode('utf-8')
        # Convert to DataFrame
        df = pd.read_csv(StringIO(data))
        dataframes.append(df)
    return dataframes

def get_csv_from_s3(s3uri, file_name):
    parsed_url = urlparse(s3uri)
    bucket_name = parsed_url.netloc
    prefix = parsed_url.path[1:].strip("/")
    s3 = boto3.resource("s3")
    obj = s3.Object(bucket_name, "{}/{}".format(prefix, file_name))
    return obj.get()["Body"].read().decode("utf-8")

def prep_rainfall(df):
    '''
    preprocess rainfall data
    '''
    df['hour'] = df['Time'].dt.hour
    df['wday'] = df['Time'].dt.dayofweek
    df['month'] = df['Time'].dt.month
    df['mday'] = df['Time'].dt.days_in_month
    df["doy"] = df['Time'].dt.dayofyear
    df['Rainlag1'] = df['Rainfall'].shift(1)
    df['Rainlag2'] = df['Rainfall'].shift(2)
    df['Rainlag1'] = df['Rainlag1'].bfill()
    df['Rainlag2'] = df['Rainlag2'].bfill()
    df['Rain_L3HR'] = df.loc[:,'Rainfall'].rolling(window=3).sum()
    df['Rain_L6HR'] = df.loc[:,'Rainfall'].rolling(window=6).sum()
    df['Rain_L12HR'] = df.loc[:,'Rainfall'].rolling(window=12).sum()
    df['Rain_L24HR'] = df.loc[:,'Rainfall'].rolling(window=24).sum()
    df['Rain_L48HR'] = df.loc[:,'Rainfall'].rolling(window=48).sum()
    df['Rain_L3HR'] = df['Rain_L3HR'].bfill()
    df['Rain_L6HR'] = df['Rain_L6HR'].bfill()
    df['Rain_L12HR'] = df['Rain_L12HR'].bfill()
    df['Rain_L24HR'] = df['Rain_L24HR'].bfill()
    df['Rain_L48HR'] = df['Rain_L48HR'].bfill()
    
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Typical value between 0.8 and 0.95
    k_api = 0.85
    
    # Initialize API and Soil Moisture as columns
    df['API'] = 0
    df['SoilMoisture'] = 0
    
    # Calculate Antecedent Precipitation Index (API)
    for i in range(1, len(df)):
        df.at[i, 'API'] = k_api * df.at[i - 1, 'API'] + df.at[i - 1, 'Rainfall']
    
    # Soil Moisture Index (very simplified: saturates at some threshold)
    # Arbitrary max bucket size (mm)
    max_storage = 100
    # Moisture decay rate
    soil_decay = 0.95
    # Initial condition
    df.at[0, 'SoilMoisture'] = min(max_storage, df.at[0, 'Rainfall'])
    
    for i in range(1, len(df)):
        df.at[i, 'SoilMoisture'] = min(
            max_storage,
            df.at[i - 1, 'SoilMoisture'] * soil_decay + df.at[i - 1, 'Rainfall']
        )

    return df



if __name__ == '__main__':
    '''
    write the script to run the transform job using defined instance type
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--instance_type', 
        type = str, 
        default = 'ml.c5.xlarge',
        help = 'aws compute instancey type'
        )
    parser.add_argument(
        '--csv_file_rf', 
        type = str, 
        default = None,
        help = 'file to process'
        )
    args, unknown = parser.parse_known_args()
    print(f"Received arguments: {args}")
    # Job setting and input file filtering
    instance_type = args.instance_type # 'ml.c5.xlarge'
    instance_count = 4
    sagemaker_model_service_worker = '2'
    csv_files_rf = [args.csv_file_rf]
    print(f"Received document: {csv_files_rf}")
    print(f"Received instance type: {instance_type}")
    # ['WWL/bias_adjusted_data/ACCESS-CM2/Pinehaven_ACCESS-CM2_ssp126_adjusted_2015_2100.csv']
    
    # Define session parameters
    region = boto3.Session().region_name
    session = sagemaker.Session()
    bucket = session.default_bucket()
    prefix = "sagemaker/autopilot-water-demand-prediction"
    role = get_execution_role()
    # This is the client we will use to interact with SageMaker Autopilot
    sm = boto3.Session().client(service_name="sagemaker", region_name=region)
    s3 = boto3.client("s3")
    fs = s3fs.S3FileSystem(anon=False)
    region = boto3.Session().region_name
    
    # Query S3 for existing csv files
    bucket_name = 'niwa-water-demand-modelling'
    key_path_rf = 'WWL/bias_adjusted_data/'
    key_path_temp = 'WWL/cmip6/'
    # csv_files_rf = list_csv_files(bucket_name, key_path_rf)
    csv_files_temp = list_csv_files(bucket_name, key_path_temp)
    
    # Extract exp_name and rf_name from each string
    temp_extracts = []
    for path in csv_files_temp:
        x = path.split('.csv')[0]
        exp_name = '_'.join(x.split('/')[-1].split('_')[1:3])  # model + scenario
        rf_name = x.split('/')[-1].split('_')[-1] # location
        rf_name = rf_name.replace('Stream', '')
        print({'exp_name': exp_name, 'rf_name': rf_name})
        temp_extracts.append({'exp_name': exp_name, 'rf_name': rf_name})
    
    rf_dict = {
        'Pinehaven': ['PCDUpperHutt'],
        'BirchLane': ['PCDStokesValley', 'PCDLowerHutt']
    }
    
    model_lst = ['CUR', 'ABI', 'ABJ', 'ABL', 'ABK']
    
    # Get completed files
    completed_files = list_csv_files(bucket, prefix+'/cmip6-result')
    completed_files = [e for e in completed_files if 'pred_data' in e]

    # Extract exp_name and rf_name from each string
    for path in csv_files_rf:
        match = re.search(r'/([^/_]+(?: [^/_]+)*)_([^/_]+_[^/_]+)_adjusted', path)
        if match:
            rf_name = match.group(1)
            rf_name = rf_name.replace(' ', '')
            exp_name = match.group(2)
            scen_name = exp_name.split('_')[-1]
            if scen_name == 'hist':
                scen_name = 'historical'
            
            # look for temp extracts info
            for i, x in enumerate(temp_extracts):
                exp_name_x = x['exp_name']
                rf_name_x = x['rf_name']
                if rf_name_x.lower() == rf_name.lower() and exp_name_x in exp_name:
                    temp_key = csv_files_temp[i]
                    print({'rf_name': rf_name, 'exp_name': exp_name, 'scen_name': scen_name})
                    print(f'temp key_file: {temp_key}')
                    break
            df_rf = read_csv_files_to_dataframes(bucket_name, [path])[0]
            df_rf.rename(columns={'time': 'Time', '0': 'Rainfall'}, inplace=True)
            df_rf = df_rf.set_index(pd.to_datetime(df_rf["Time"], format="%Y-%m-%d %H:%M:%S"))
            df_rf = df_rf.resample('h').sum(['Rainfall'])
            df_temp = read_csv_files_to_dataframes(bucket_name, [temp_key])[0]
            df_temp['Dry bulb degC'] = df_temp[scen_name] - 273
            df_temp = df_temp[['date', 'Dry bulb degC']]
            df_temp['date'] = pd.to_datetime(df_temp['date'], format="%Y-%m-%d %H:%M:%S")
            df_temp = df_temp.set_index('date')
    
            df_rf_1 = prep_rainfall(df_rf[['Rainfall']].reset_index())
            df = df_rf_1.set_index('Time').join(df_temp['Dry bulb degC'])
            df = df.reset_index()
    
            # find suitable job names
            job_names = []
            job_prefixes = rf_dict[rf_name]
            for job_prefix in job_prefixes:
                for job_suffix in model_lst:
                    job_name = f'{job_prefix}{job_suffix}'
                    completed_file_key = f'{rf_name}_{exp_name}_{job_name}'
                    completed_file_found = [e for e in completed_files if completed_file_key in e]
                    if len(completed_file_found)>0:
                        print(f'{completed_file_key} found, skip job creation')
                        continue
                    job_names.append(job_name)
            print("all job names: ", job_names)

            columns = [e for e in df.columns if e not in ["Time", "sin_hour", "cos_hour"]]
            test_data = df[columns] # Features
            test_file = f"{rf_name}_{exp_name}.csv"
            test_data.to_csv(test_file, index=False, header=False)
            test_data_s3_path = session.upload_data(path=test_file, key_prefix=prefix)
            print("Test data uploaded to: " + test_data_s3_path)
            
            for job_name in job_names:
                print("running job: ", job_name)
                # This is the client we will use to interact with SageMaker Autopilot
                sm = boto3.Session().client(service_name="sagemaker", region_name=region)
                auto_ml_job_name = f"automl-{job_name}"
                print("AutoMLJobName: " + auto_ml_job_name)
                best_candidate = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)["BestCandidate"]
                best_candidate_name = best_candidate["CandidateName"]
                
                print("\n")
                print("CandidateName: " + best_candidate_name)
                print(
                    "FinalAutoMLJobObjectiveMetricName: "
                    + best_candidate["FinalAutoMLJobObjectiveMetric"]["MetricName"]
                )
                print(
                    "FinalAutoMLJobObjectiveMetricValue: "
                    + str(best_candidate["FinalAutoMLJobObjectiveMetric"]["Value"])
                )
            
                automl = AutoML.attach(auto_ml_job_name=auto_ml_job_name)
                
                s3_transform_output_path = "s3://{}/{}/cmip6-results/".format(bucket, prefix)
                
                model_name = "{0}-model".format(best_candidate_name)
                
                model = automl.create_model(
                    name=model_name,
                    candidate=best_candidate,
                )
                
                output_path = s3_transform_output_path + best_candidate_name + "/"
                
                transformer = model.transformer(
                    instance_count=instance_count,
                    instance_type=instance_type,
                    assemble_with="Line",
                    strategy="SingleRecord",
                    output_path=output_path,
                    env={"SAGEMAKER_MODEL_SERVER_TIMEOUT": "100", "SAGEMAKER_MODEL_SERVER_WORKERS": sagemaker_model_service_worker},
                )
            
                transformer.transform(
                    data=test_data_s3_path,
                    split_type="Line",
                    content_type="text/csv",
                    wait=False,
                    model_client_config={"InvocationsTimeoutInSeconds": 80, "InvocationsMaxRetries": 1},
                )
                
                print("Starting transform job {}".format(transformer._current_job_name))
            
                ## Wait for jobs to finish
                pending_complete = True
                batch_job_name = transformer._current_job_name
                
                while pending_complete:
                    pending_complete = False
                
                    description = sm.describe_transform_job(TransformJobName=batch_job_name)
                    if description["TransformJobStatus"] not in ["Failed", "Completed"]:
                        pending_complete = True
                
                    print("{} transform job is running.".format(batch_job_name))
                    time.sleep(60)
                
                print("\nCompleted.")
                
                job_status = sm.describe_transform_job(TransformJobName=batch_job_name)["TransformJobStatus"]
                
                if job_status == "Completed":
                    pred_csv = get_csv_from_s3(transformer.output_path, "{}.out".format(test_file))
                    predictions = pd.read_csv(io.StringIO(pred_csv), header=None)
        
                    df[job_name] = predictions
                    df[job_name] = np.where(df[job_name] < 0, 0, df[job_name])
                    pred_to_move = df.pop(job_name)
                    df.insert(1, pred_to_move.name, pred_to_move)
            
                ### Upload the dataset to S3
                pred_file = f"{rf_name}_{exp_name}_{job_name}_pred_data.csv"
                df[['Time', job_name]].to_csv(f'cmip6/{pred_file}', index=False, header=True)
                pred_data_s3_path = session.upload_data(path=f'cmip6/{pred_file}', key_prefix=prefix + f"/cmip6-results")
                print("Full pred results uploaded to: " + pred_data_s3_path)
