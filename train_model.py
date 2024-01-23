import joblib
from data_loading import load_and_prepare_data, convert_time_to_minutes
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_and_evaluate_model_with_cv(df):
    df_for_training = df.copy()

    # Convert the 'time' column to minutes since midnight
    if 'time' in df_for_training.columns:
        df_for_training['time'] = df_for_training['time'].apply(convert_time_to_minutes)

    # Data preparation for training
    X = df_for_training.drop('status', axis=1)
    y = df_for_training['status']

    # Model
    model = RandomForestClassifier(random_state=42)

    # Apply cross validation
    scores = cross_val_score(model, X, y, cv=5)

    # Train the model with the entire dataset
    model.fit(X, y)

    # Print cross-validation results
    print(f"Resultados da Validação Cruzada: {scores}")
    print(f"Média dos Resultados: {scores.mean()}")

    return model

def train_and_evaluate_model(df):
    df_for_training = df.copy()

    # Check if the 'time' column exists and convert it to minutes since midnight
    if 'time' in df_for_training.columns:
        df_for_training['time'] = df_for_training['time'].apply(convert_time_to_minutes)

    # Data preparation for training
    X = df_for_training.drop('status', axis=1)
    y = df_for_training['status']

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return model







