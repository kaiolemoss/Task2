from data_loading import load_and_prepare_data
from data_analysis import analyze_transactions, detect_anomalies
from real_time_visualization import plot_transactions, plot_f1_scores, plot_confusion_matrix
from train_model import train_and_evaluate_model_with_cv, train_and_evaluate_model
import joblib

def run_analysis(file_path1, file_path2):
    # Load and prepare data
    df1, label_encoders1 = load_and_prepare_data(file_path1)
    df2, label_encoders2 = load_and_prepare_data(file_path2)

    # Train and evaluate the model
    model1 = train_and_evaluate_model(df1)
    model2 = train_and_evaluate_model(df2)
    # Save the trained model
    joblib.dump(model1, 'transaction_model1.pkl')
    joblib.dump(model2, 'transaction_model2.pkl')

    # Train and evaluate the model with cross-validation
    model1_cv = train_and_evaluate_model_with_cv(df1)
    model2_cv = train_and_evaluate_model_with_cv(df2)
    # Save the trained model
    joblib.dump(model1_cv, 'transaction_model1_cv.pkl')
    joblib.dump(model2_cv, 'transaction_model2_cv.pkl')

    # Performing analysis with the model
    analysis_results1 = analyze_transactions(df1, label_encoders1, model1)
    analysis_results2 = analyze_transactions(df2, label_encoders2, model2)

    # Performing data analysis
    anomalies1 = detect_anomalies(analysis_results1)
    anomalies2 = detect_anomalies(analysis_results2)

    # Print analysis results
    print("Análise de Transações 1:", analysis_results1)
    print("Anomalias em Transações 1:", anomalies1)
    print("Análise de Transações 2:", analysis_results2)
    print("Anomalias em Transações 2:", anomalies2)    

    classes = ['0', '1', '2', '3', '4', '5', '6']

    #transaction_1
    f1_scores_1 = [0.93, 0.24, 0.85, 0.27, 0.34, 0.25, 0.47]
    confusion_matrix_values_1 = [
        [245, 1, 14, 0, 1, 1, 6],
        [0, 18, 5, 0, 10, 28, 12],
        [11, 2, 217, 1, 1, 2, 11],
        [0, 2, 0, 3, 2, 5, 0],
        [0, 6, 4, 1, 11, 10, 2],
        [0, 20, 3, 3, 5, 21, 20],
        [5, 29, 20, 2, 1, 26, 60]
    ]

    #transactions_2
    f1_scores_2 = [0.96, 0.00, 0.92, 0.83, 0.00, 0.40, 0.57]
    confusion_matrix_values_2 = [
        [260, 0, 9, 0, 0, 0, 2],
        [1, 0, 3, 0, 1, 13, 7],
        [9, 0, 204, 0, 3, 1, 6],
        [0, 0, 1, 30, 0, 1, 4],
        [1, 1, 0, 0, 0, 0, 2],
        [0, 6, 0, 0, 1, 34, 37],
        [2, 12, 4, 6, 1, 44, 83]
    ]

    # Call plotting functions
    #plot_transactions(df1, label_encoders1)
    #plot_transactions(df2, label_encoders2)
    plot_f1_scores(classes, f1_scores_1)
    plot_confusion_matrix(confusion_matrix_values_1)
    plot_f1_scores(classes, f1_scores_2)
    plot_confusion_matrix(confusion_matrix_values_2)

    
    return df1, df2, label_encoders1, label_encoders2, analysis_results1, anomalies1, analysis_results2, anomalies2, model1, model2

if __name__ == '__main__':
    file_path1 = 'transactions_1.csv'
    file_path2 = 'transactions_2.csv'
    run_analysis(file_path1, file_path2)
