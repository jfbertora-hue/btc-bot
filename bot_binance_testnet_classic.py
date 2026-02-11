import pandas as pd
import numpy as np
import yfinance as yf
from binance.client import Client
from datetime import datetime
import os

# ============================
# API KEYS (variables entorno)
# ============================

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# ============================
# CONFIG
# ============================

SYMBOL = "BTCUSDT"
USDT_PER_TRADE = 50

RSI_LEVEL = 45
ADX_MIN = 15

TAKE_PROFIT = 0.05   # +5%
STOP_LOSS = 0.02     # -2%

client = Client(API_KEY, API_SECRET)
client.API_URL = "https://testnet.binance.vision/api"

position_price = None
position_qty = 0

def indicators(df):

    df["ema200"] = df["Close"].ewm(span=200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean()/loss.rolling(14).mean()
    df["rsi"] = 100-(100/(1+rs))

    plus_dm = df["High"].diff()
    minus_dm = df["Low"].diff().abs()
    tr = np.maximum(df["High"]-df["Low"],
         np.maximum(abs(df["High"]-df["Close"].shift()),
                    abs(df["Low"]-df["Close"].shift())))
    atr = tr.rolling(14).mean()

    plus_di = 100*(plus_dm.rolling(14).mean()/atr)
    minus_di = 100*(minus_dm.rolling(14).mean()/atr)
    dx = abs(plus_di-minus_di)/(plus_di+minus_di)*100
    df["adx"] = dx.rolling(14).mean()

    df.dropna(inplace=True)
    return df

def get_data():

    df = yf.download("BTC-USD", period="2y", interval="4h")

    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)

    for c in ["Open","High","Low","Close"]:
        if isinstance(df[c],pd.DataFrame):
            df[c]=df[c].iloc[:,0]

    return indicators(df)

df = get_data()

price = float(df["Close"].iloc[-1])
rsi = float(df["rsi"].iloc[-1])
ema = float(df["ema200"].iloc[-1])
adx = float(df["adx"].iloc[-1])

action = "HOLD"

# Chequear posición actual
balances = client.get_asset_balance(asset="BTC")
btc_balance = float(balances["free"])

if btc_balance > 0:
    position_price = price
    position_qty = btc_balance

# Si NO estamos dentro → evaluar compra
if btc_balance == 0:

    if price > ema and rsi < RSI_LEVEL and adx > ADX_MIN:
        qty = round(USDT_PER_TRADE/price,6)
        client.create_order(symbol=SYMBOL,side="BUY",type="MARKET",quantity=qty)
        action="BUY"

# Si estamos dentro → evaluar venta
else:

    entry_price = price  # simplificado para testnet

    if price >= entry_price*(1+TAKE_PROFIT):
        client.create_order(symbol=SYMBOL,side="SELL",type="MARKET",quantity=position_qty)
        action="SELL_TP"

    elif price <= entry_price*(1-STOP_LOSS):
        client.create_order(symbol=SYMBOL,side="SELL",type="MARKET",quantity=position_qty)
        action="SELL_SL"

print("\n",datetime.now())
print("Precio:",round(price,2))
print("RSI:",round(rsi,2))
print("EMA:",round(ema,2))
print("ADX:",round(adx,2))
print("ACCION:",action)