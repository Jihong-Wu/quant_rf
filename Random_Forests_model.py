import pandas as pd
import numpy as np
from datetime import datetime, time
import matplotlib.pyplot as plt
import indicators as ti
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score


def features_extraction(data): # extract features from raw data
    data_copy = data.reset_index(drop=True).copy()
    
    for x in [2,5,10]:
        data_copy = ti.relative_strength_index(data_copy, x)
        data_copy = ti.average_true_range(data_copy, x)
        data_copy = ti.stochastic_oscillator(data_copy, x)
        data_copy = ti.accumulation_distribution(data_copy, x)
        data_copy = ti.momentum(data_copy, x)
        data_copy = ti.rate_of_change(data_copy, x)
        data_copy = ti.on_balance_volume(data_copy, x)
        data_copy = ti.commodity_channel_index(data_copy, x)
        data_copy = ti.trix(data_copy, x)
        data_copy['ema_' + str(x)] = data_copy['Average']/data_copy['Average'].ewm(span=x, adjust=False).mean()
    data_copy = ti.ease_of_movement(data_copy)
    data_copy = ti.macd(data_copy,n_fast=5,n_slow=20)
    data_copy.index = data.index[:len(data_copy)]
    del(data_copy['max'])
    del(data_copy['min'])
    del(data_copy['Endprice'])
    return data_copy


def prepare_data(bars, horizon):
    # make it ternary classification problem
    # up=1, down=-1, neutral=0
    threshold = 0.001
    r = bars['Average'].shift(-horizon) - bars['Average']
    pred = np.where(r > threshold, 2, np.where(r < -threshold, 0, 1))
    
    # Create a full-length array with NaN for the last 'horizon' rows
    full_pred = np.full(len(bars), np.nan)
    full_pred[:-horizon] = pred[:-horizon]
    
    return pd.Series(full_pred, index=bars.index, dtype='Int64')  # Use nullable integer type

if __name__ == "__main__":
    df = pd.read_csv("SIG/DataExercise.csv")
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["TimeOfDay"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "timestamp"]).reset_index(drop=True)
    df["slice_start"] = df["timestamp"].dt.floor("10min")  # 10 minutes interval
    df["slice_label"] = (
        df["slice_start"].dt.strftime("%H:%M") + "–" +
        (df["slice_start"] + pd.Timedelta("10min")).dt.strftime("%H:%M")
    )
    # delete rows outside 15:30
    df = df[(df["timestamp"].dt.time >= time(10, 0)) & (df["timestamp"].dt.time <= time(15, 30))].reset_index(drop=True)
    # Then calculate features of slices
    df = df.set_index('timestamp')
    bars = df.groupby('slice_start').agg({'Spot': ['max', 'min']}).droplevel(0, axis=1)
    bars['Average'] = df['Spot'].resample('10min').mean()
    bars["Endprice"]= df['Spot'].resample('10min').last()
    bars['Range'] = bars['max'] - bars['min']
    bars['Drift']= df['Spot'].resample('10min').last() - df['Spot'].resample('10min').first()
    bars['Volume'] = df['Spot'].resample('10min').size().clip(lower=1)
    bars['Std']= df['Spot'].resample('10min').std().fillna(0)
    bars['S']=bars['Range']/bars['Volume'].pow(0.5).dropna()

    bars['TradeCount'] = df['Spot'].resample('10min').count()
    bars['TradeCount_Fast'] = bars['TradeCount'].rolling(window=3, min_periods=1).mean()
    bars['TradeCount_Slow'] = bars['TradeCount'].rolling(window=10, min_periods=1).mean()
    bars['TradeCount_Diff'] = bars['TradeCount_Fast'] - bars['TradeCount_Slow']


    bars['UpTrends'] = df.resample('10min').apply(ti.count_up_trends).fillna(0)
    bars['DownTrends'] = df.resample('10min').apply(ti.count_down_trends).fillna(0)
    bars['TrendDiff'] = bars['UpTrends'] - bars['DownTrends']
    del(bars['UpTrends'])
    del(bars['DownTrends'])


    # Prepare data with a prediction horizon of next slice
    horizon = 1
    bars['Prediction'] = prepare_data(bars, horizon)
    
    # Apply feature extraction and extract y from the same cleaned data
    data = features_extraction(bars).dropna()
    y = data['Prediction'].copy()  # Extract y from cleaned data before reset_index
    
    # Now reset index for both to maintain alignment
    data = data.reset_index()
    y = y.reset_index(drop=True)
    features = [x for x in data.columns if x not in ['slice_start', 'Prediction']]
    x = data[features]
    x = x.replace([np.inf, -np.inf], np.nan)
    for col in x.columns:
        if x[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Cap values at 99.9th percentile to handle outliers
            upper_limit = x[col].quantile(0.999)
            lower_limit = x[col].quantile(0.001)
            x[col] = x[col].clip(lower=lower_limit, upper=upper_limit)
    before_dropna = len(x)
    valid_mask = ~x.isnull().any(axis=1)
    x = x[valid_mask]
    y = y[valid_mask]
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Create train/test split based on cleaned data indices
    split_idx = 9*len(x) // 10
    X_train = x.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = x.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    y_train = y_train.astype('int').values.ravel()
    y_test = y_test.astype('int').values.ravel()

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Fix class distribution display for labels 0, 1, 2
unique_train, counts_train = np.unique(y_train, return_counts=True)
class_dist_train = dict(zip(unique_train, counts_train))
class_dist_train = dict(zip(unique_train, counts_train))

unique_test, counts_test = np.unique(y_test, return_counts=True)
class_dist_test = dict(zip(unique_test, counts_test))


# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, max_features="sqrt", 
                           n_jobs=-1, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)


# Show prediction distribution
y_pred = rf.predict(X_test)  # Predict on the entire test set
unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
class_dist_pred = dict(zip(unique_pred, counts_pred))
print(f"Prediction distribution: {class_dist_pred}")

# Evaluate the model with zero_division parameter
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("Recall Score:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
print("Precision Score:", precision_score(y_test, y_pred, average='weighted', zero_division=0))

# Show feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
