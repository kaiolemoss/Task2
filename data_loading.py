import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    # Convert 'time' column to time objects - ISO8601
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], format='%H:%M').dt.time

    #Transform non-numeric "status" labels into numbers to apply ML
    label_encoders = {} 
    for column in df.columns:
        if df[column].dtype == object and column != 'time':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    
    return df, label_encoders

def inspect_data(df):
    info = df.info()
    head = df.head()
    return info, head

def convert_time_to_minutes(time_val):
    if time_val is not None:
        return time_val.hour * 60 + time_val.minute
    else:
        return None

