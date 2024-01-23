import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_transactions(df, label_encoders):
    # Get the numeric code for 'failed'
    failed_code = label_encoders['status'].transform(['failed'])[0]

    # Check for failed transactions
    failed_transactions = df[df['status'] == failed_code]
    if failed_transactions.empty:
        print("Não há transações com status 'failed' para plotar.")
        return

    # Convert the 'time' column to date/time format, ignoring invalid values
    df['time'] = pd.to_datetime(df['time'], format='%H:%M', errors='coerce').dt.time

    # Filter null data
    df = df[df['time'].notnull()]

    # Plot the number of failed transactions over time
    plt.figure(figsize=(10, 6))
    failed_transactions.groupby('time').size().plot()
    plt.title('Failed Transactions Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Failed Transactions')
    plt.show()

def plot_f1_scores(classes, f1_scores):
    plt.bar(classes, f1_scores, color='blue')
    plt.xlabel('Classes')
    plt.ylabel('F1-Score')
    plt.title('F1-Score por Classe')
    plt.show()

def plot_confusion_matrix(confusion_matrix):
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Heatmap')
    plt.show()





