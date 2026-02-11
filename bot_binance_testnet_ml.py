import pandas as pd
import numpy as np
import yfinance as yf
from binance.client import Client
from datetime import datetime
from ml_trend_filter import predict_signal
import os

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

SYMBOL="BTCUSDT"
USDT_PER_TRADE=50
RSI_LEVEL=45
ADX_MIN=15

client = Client(API_KEY,API_SECRET)
client.API_URL="https://testnet.binance.vision/api"

def indicators(df):
    df["ema"]=df["Close"].ewm(span=200).mean()

    delta=df["Close"].diff()
    gain=delta.clip(lower=0)
    loss=-delta.clip(upper=0)
    rs=gain.rolling(14).mean()/loss.rolling(14).mean()
    df["rsi"]=100-(100/(1+rs))

    tr=np.maximum(df["High"]-df["Low"],
         np.maximum(abs(df["High"]-df["Close"].shift()),
         abs(df["Low"]-df["Close"].shift())))

    atr=tr.rolling(14).mean()
    plus= df["High"].diff()
    minus= df["Low"].diff().abs()
    dx=abs(plus-minus)/(plus+minus)*100
    df["adx"]=dx.rolling(14).mean()

    df.dropna(inplace=True)
    return df

def get_data():
    df=yf.download("BTC-USD",period="2y",interval="4h")
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)
    return indicators(df)

df=get_data()

price=float(df["Close"].iloc[-1])
rsi=float(df["rsi"].iloc[-1])
ema=float(df["ema"].iloc[-1])
adx=float(df["adx"].iloc[-1])

prob,ret=predict_signal(df)

btc=float(client.get_asset_balance(asset="BTC")["free"])

action="HOLD"

if price>ema and rsi<RSI_LEVEL and adx>ADX_MIN and prob>0.6 and ret>0:
    qty=round(USDT_PER_TRADE/price,6)
    client.create_order(symbol=SYMBOL,side="BUY",type="MARKET",quantity=qty)
    action="BUY"

elif btc>0 and (ret<0 or prob<0.4):
    client.create_order(symbol=SYMBOL,side="SELL",type="MARKET",quantity=round(btc,6))
    action="SELL"

print("\n",datetime.now())
print("Precio:",round(price,2))
print("RSI:",round(rsi,2))
print("EMA:",round(ema,2))
print("ADX:",round(adx,2))
print("ML prob:",round(prob,2))
print("ML ret:",round(ret,4))
print("ACCION:",action)
