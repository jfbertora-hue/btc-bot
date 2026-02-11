import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
FEATURE_FILE = "features.pkl"

FEATURES = ["rsi","body","upper_wick","lower_wick","ret"]

# ============================
# FEATURE BUILDER
# ============================

def build_features(df):

    df["ret"] = df["Close"].pct_change()
    df["ema200"] = df["Close"].ewm(span=200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + rs))

    df["body"] = abs(df["Close"] - df["Open"])
    df["upper_wick"] = df["High"] - df[["Close","Open"]].max(axis=1)
    df["lower_wick"] = df[["Close","Open"]].min(axis=1) - df["Low"]

    df.dropna(inplace=True)
    return df

# ============================
# TRAIN
# ============================

def train_model():

    print("Descargando BTC diario desde 2016...")

    df = yf.download("BTC-USD", start="2016-01-01", interval="1d")

    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)

    for c in ["Open","High","Low","Close"]:
        if isinstance(df[c],pd.DataFrame):
            df[c]=df[c].iloc[:,0]

    df = build_features(df)

    df["ema200"] = df["Close"].ewm(span=200).mean()
    df["target"] = (df["Close"] > df["ema200"]).astype(int)

    X = df[FEATURES]
    y = df["target"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=400,max_depth=7,random_state=42)
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    print("\nACCURACY:",model.score(X_test,y_test))
    print("ROC:",roc_auc_score(y_test,prob))
    print(classification_report(y_test,pred))

    joblib.dump(model,MODEL_FILE)
    joblib.dump(scaler,SCALER_FILE)
    joblib.dump(FEATURES,FEATURE_FILE)

    print("\nMODELOS GUARDADOS")

# ============================
# PREDICT
# ============================

def predict_signal(df):

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    df = build_features(df)

    X = df[FEATURES].iloc[-1:]
    X = scaler.transform(X)

    prob = model.predict_proba(X)[0][1]

    ret = df["ret"].iloc[-1]

    return float(prob), float(ret)

# ============================
# MAIN TRAIN CALL
# ============================

if __name__ == "__main__":

    if not os.path.exists(MODEL_FILE):
        train_model()
    else:
        print("Modelo ya existe — salteando entrenamiento.")
