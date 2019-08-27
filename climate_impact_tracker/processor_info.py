import requests
import pandas as pd
from subprocess import Popen, PIPE

from xml.etree.ElementTree import fromstring

import cpuinfo
from bs4 import BeautifulSoup
import re
import os

def get_gpu_info():
    p = Popen(['nvidia-smi', '-q', '-x'], stdout=PIPE)
    outs, errors = p.communicate()
    xml = fromstring(outs)
    datas = []
    for gpu_id, gpu in enumerate(xml.getiterator('gpu')):
        gpu_data = {}
        name = [x for x in gpu.getiterator('product_name')][0].text
        gpu_data['name'] = name
        datas.append(gpu_data)
    return datas

def get_my_cpu_info():
    return cpuinfo.get_cpu_info()

def get_and_cache_cpu_max_tdp_from_intel():
    # Realized this isn't really worth anything because TDP isn't a reliable estimator of power output
    cpu_brand = cpuinfo.get_cpu_info()['brand'].split(' ')[2]
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpuinfocache/{}'.format(cpu_brand))):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpuinfocache/{}'.format(cpu_brand)), 'r') as f:
            return int(f.readline())
    s = requests.Session()
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36'
    s.headers['User-Agent'] = user_agent
    r = s.get('https://ark.intel.com/content/www/us/en/ark/search.html?_charset_=UTF-8&q={}'.format(cpu_brand),allow_redirects=True)
    soup= BeautifulSoup(r.content,'lxml')
    results = soup.find_all('span', attrs={'data-key' : "MaxTDP"})

    if len(results) == 0:
        redirect_url = soup.find(id='FormRedirectUrl').attrs['value']
        if redirect_url:
            r = s.get("https://ark.intel.com/" + redirect_url, allow_redirects=True)
            soup= BeautifulSoup(r.content,'lxml')
            results = soup.find_all('span', attrs={'data-key' : "MaxTDP"})

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpuinfocache/{}'.format(cpu_brand)), 'w') as f:
        f.write((results[0].text.strip().replace('W','')))
    return int(results[0].text.strip().replace('W',''))
