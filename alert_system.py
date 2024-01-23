from flask import Flask, request, jsonify
from data_loading import load_and_prepare_data
from data_analysis import analyze_transactions, detect_anomalies
from anomaly_reporting import report_anomaly

app = Flask(__name__)

# Store data from the last processed transaction
last_transaction_data = {}

@app.route('/process_transaction', methods=['POST'])
def process_transaction():
    global last_transaction_data
    transaction_data = request.json
    last_transaction_data = transaction_data  # Store transaction data
    df = load_and_prepare_data(transaction_data)
    analysis_results = analyze_transactions(df)
    anomalies = detect_anomalies(df)
    if anomalies:
        report_anomaly(anomalies)
        return jsonify({'decision': 'alert'})
    return jsonify({'decision': 'no alert'})

@app.route('/last_transaction', methods=['GET'])
def get_last_transaction():
    if last_transaction_data:
        return jsonify({'last_transaction': last_transaction_data}), 200
    else:
        return "No transactions processed yet.", 404

@app.route('/', methods=['GET'])
def home():
    return "Alert System is Running!", 200

def run_server():
    app.run(debug=True)
