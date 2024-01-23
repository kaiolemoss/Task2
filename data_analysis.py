from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_loading import convert_time_to_minutes
import pandas as pd

def analyze_transactions(df, label_encoders, model):
    # Get the numeric codes for the relevant categories
    failed_code = label_encoders['status'].transform(['failed'])[0]
    denied_code = label_encoders['status'].transform(['denied'])[0]
    reversed_code = label_encoders['status'].transform(['reversed'])[0]
    refunded_code = label_encoders['status'].transform(['refunded'])[0]
    backend_reversed_code = label_encoders['status'].transform(['backend_reversed'])[0]

    results = {
        'failed_transactions': df[df['status'] == failed_code].shape[0],
        'denied_transactions': df[df['status'] == denied_code].shape[0],
        'reversed_transactions': df[df['status'] == reversed_code].shape[0],
        'refunded_transactions': df[df['status'] == refunded_code].shape[0],
        'backend_reversed_transactions': df[df['status'] == backend_reversed_code].shape[0]
    }

    # Add a new column with the anomaly score
    df['anomaly_score'] = df.apply(lambda row: score_anomaly(model, row, label_encoders), axis=1)

    if 'time' in df.columns:
            df['hour'] = df['time'].apply(lambda x: x.hour if x is not None else None)
            hourly_analysis = df.groupby('hour').size()
            results['hourly_transactions'] = hourly_analysis.to_dict()    

    print(df['time'].head()) # Check the first values ​​in the 'time' column
    df['hour'] = df['time'].apply(lambda x: x.hour if x is not None else None)
    print(df['hour'].head())  # Check the first values ​​of the 'hour' column after conversion
    hourly_analysis = df.groupby('hour').size()
    print(hourly_analysis)  

    return results

def detect_anomalies(analysis_results, threshold_failed=1.5, threshold_denied=1.5, threshold_reversed=1.5, threshold_refunded=1.5, threshold_backend_reversed=1.5):
    anomalies = {}
    if (analysis_results['failed_transactions'] > threshold_failed or
        analysis_results['denied_transactions'] > threshold_denied or
        analysis_results['reversed_transactions'] > threshold_reversed or
        analysis_results['refunded_transactions'] > threshold_refunded or
        analysis_results['backend_reversed_transactions'] > threshold_backend_reversed):
        anomalies['failed'] = True
    return anomalies

def score_anomaly(model, transaction, label_encoders):
    # Copy the transaction so as not to change the original DataFrame
    transaction = transaction.copy()

    # Convert the 'time' column to minutes since midnight
    if 'time' in transaction and transaction['time'] is not None:
        transaction['time'] = transaction['time'].hour * 60 + transaction['time'].minute

    # Remove the 'status' column before making the prediction
    if 'status' in transaction:
        transaction.pop('status')

    # Transform the transaction into a DataFrame
    features = pd.DataFrame([transaction])
    
    # Encode categorical variables if necessary
    for col in features.columns:
        if features[col].dtype == object and col in label_encoders:
            features[col] = label_encoders[col].transform(features[col])

    # Calculate class probabilities
    probabilities = model.predict_proba(features)
    # Returns the probability of the anomalous class (assuming it is the second class)
    return probabilities[0][1]




