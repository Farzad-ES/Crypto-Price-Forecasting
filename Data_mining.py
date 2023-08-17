import pandas as pd
import datetime
import time
import requests
import hmac
import os
import json
from dotenv.main import load_dotenv
from hashlib import sha256

load_dotenv("./dev.env")
api_url="https://open-api.bingx.com"
api_key=os.getenv("api_key")
secret_key=os.getenv("secret_key")

def data_mining(crypto_pair, interval, start_time, end_time):
    payload = {}
    path = '/openApi/swap/v2/quote/klines'
    method = "GET"
    intervals={
        "1m":86400000,
        "3m":259200000,
        "5m":432000000,
        "15m":1296000000,
        "30m":2592000000,
        "1h":5184000000,
        "2h":10368000000,
        "4h":20736000000,
        "6h":31104000000,
        "8h":41472000000,
        "12h":62208000000,
        "1d":124416000000,
        "3d":373248000000,
        "1w":870912000000,
        "1M":3732480000000
    }
    start_timestamp=int(time.mktime(datetime.datetime.strptime(start_time,"%Y/%m/%d").timetuple())*1000)
    end_timestamp=int(time.mktime(datetime.datetime.strptime(end_time,"%Y/%m/%d").timetuple())*1000)
    timestamps=[]
    data=[]
    for i in range(start_timestamp, end_timestamp+1, intervals[interval]):
        timestamps.append(i)
    
    for j in range(len(timestamps)-1):
        paramsMap = {
            "symbol": crypto_pair,
            "interval": interval,
            "startTime": timestamps[j],
            "endTime": timestamps[j+1],
            "limit": 1440
        }
        paramsStr = praseParam(paramsMap)
        data_dict=json.loads(send_request(method, path, paramsStr, payload))['data']
        data_frame=pd.DataFrame(data_dict)
        data.append(data_frame)
    if timestamps[len(timestamps)-1]!=end_timestamp:
        paramsMap = {
            "symbol": crypto_pair,
            "interval": interval,
            "startTime": timestamps[len(timestamps)-1],
            "endTime": end_timestamp,
            "limit": 1440
        }
        paramsStr = praseParam(paramsMap)
        data_dict=json.loads(send_request(method, path, paramsStr, payload))['data']
        data_frame=pd.DataFrame(data_dict)
        data.append(data_frame)
    return data_preprocessing(data)
    
def get_sign(api_secret, payload):
    signature = hmac.new(api_secret.encode("utf-8"), payload.encode("utf-8"), digestmod=sha256).hexdigest()
    return signature


def send_request(method, path, urlpa, payload):
    url = "%s%s?%s&signature=%s" % (api_url, path, urlpa, get_sign(secret_key, urlpa))
    headers = {
        'X-BX-APIKEY': api_key,
    }
    response = requests.request(method, url, headers=headers, data=payload)
    return response.text

def praseParam(paramsMap):
    sortedKeys = sorted(paramsMap)
    paramsStr = "&".join(["%s=%s" % (x, paramsMap[x]) for x in sortedKeys])
    return paramsStr+"&timestamp="+str(int(time.time() * 1000))

def milisec_to_sec(x):
    return x/1000

def data_preprocessing(data_list):
    df=pd.concat(data_list, ignore_index=True)
    df['time']=df['time'].apply(milisec_to_sec)
    df['time']=df['time'].apply(datetime.datetime.fromtimestamp)
    df=df.set_index('time')
    return df