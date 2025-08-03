import numpy as np  # For RMSE calculation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_stats_summary():
    from data_processing import (
        load_and_prepare_data,
        group_by_opponent,
        calculate_team_stats,
        calculate_filter_stats
    )

    combined_data = load_and_prepare_data()

    grouped_active, _ = group_by_opponent(combined_data)

    # Define the columns to calculate cumulative stats for
    stat_columns = [
        "PTS", "AST", "TRB", "3P", "STL", "BLK"
    ]

    # Calculate team stats
    stats_summary = calculate_team_stats(grouped_active, stat_columns)

    # Enhance stats_summary with stat metrics (e.g., Avg PTS + AST, Avg PTS + REB)
    stats_summary = calculate_filter_stats(stats_summary)

    return stats_summary


# Define Prediction Process
def predict_future_stats(opponent_acronym):
    stats_summary = load_stats_summary()

    # Select features and target for the model
    feature_columns = [
        "Average PTS", "Average AST", "Average TRB",
        "Average 3P", "Average STL", "Average BLK",
        "Avg PTS + AST", "Avg PTS + REB"
    ]
    target_columns = [
        "Average PTS", "Average AST", "Average TRB",
        "Average 3P", "Average STL", "Average BLK",
        "Avg PTS + AST", "Avg PTS + REB"
    ]

    # Drop any invalid rows (if necessary)
    stats_summary = stats_summary.dropna()

    # Split data into features (X) and target (y)
    X = stats_summary[feature_columns]
    y = stats_summary[target_columns]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the feature data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Regression Model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Calculate regression evaluation metrics on test data
    predictions = model.predict(X_test_scaled)

    # Mean Squared Error
    mse = mean_squared_error(y_test, predictions)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = mean_absolute_error(y_test, predictions)


    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')

    # Calculate the baseline predictions using the mean of training target values
    baseline_predictions = np.tile(y_train.mean(axis=0), (len(y_test), 1))

    # Calculate RMSE for the baseline
    baseline_mse = mean_squared_error(y_test, baseline_predictions)
    baseline_rmse = np.sqrt(baseline_mse)

    # Calculate MAE for the baseline
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)


    print(f"Baseline Root Mean Squared Error (RMSE): {baseline_rmse:.4f}")
    print(f"Baseline Mean Absolute Error (MAE): {baseline_mae:.4f}")

    # Filter for historical data of the given opponent
    new_opponent_stats = stats_summary[stats_summary["Opponent"] == opponent_acronym]

    if new_opponent_stats.empty:
        return None

    new_opponent_stats = new_opponent_stats[feature_columns]

    # Scale the feature data
    new_opponent_stats_scaled = scaler.transform(new_opponent_stats)

    # Make predictions for the specified opponent
    future_prediction = model.predict(new_opponent_stats_scaled)

    return model, scaler, predictions, future_prediction
