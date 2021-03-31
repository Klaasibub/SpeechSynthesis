from datetime import datetime
from time import time, sleep
import requests
import json


while True:
    sleep(5)
    try:
        r = requests.get('http://localhost:4040/api/tunnels')
        print(json.loads(r.content)['tunnels'][0]['public_url'])
    except Exception as ex:
        print('err with port 4040:', ex)

    print(datetime.now())
    sleep(60)
