from main_analysis import run_analysis
from alert_system import run_server

file_path1 = 'transactions_1.csv'
file_path2 = 'transactions_2.csv'

run_analysis(file_path1, file_path2)
run_server()

