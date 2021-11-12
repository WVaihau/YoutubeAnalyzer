# -*- coding: utf-8 -*-
"""
@author: WILLIAMU Vaihau
"""

#Web
import aiohttp
import asyncio
import requests

# WebScraping
from bs4 import BeautifulSoup

# DataFrame & others
import pandas as pd
import numpy as np
import isodate
from time import time

class SampleSize(object):
    def __init__(self, N, confidence_level, margin_error=.05, std=.5, treshold=1000):
        self.zscoredict={
        "80%" : 1.28,
        "85%" : 1.44,
        "90%" : 1.65,
        "95%" : 1.96,
        "99%" : 2.58
        }
        self.pop_size = N
        self.z = self.zscoredict[confidence_level]
        self.std = std
        self.e = margin_error
        self.treshold = treshold
        self.samplesize = self.__get_samplesize()
    
    def __short_pop(self):
        sample_size = ((self.z**2 * self.std * (1 - self.std)) / (self.e**2)) / (1 + ((self.z**2 * self.std * (1 - self.std)) / (self.e**2 * self.pop_size)))
        return sample_size
    
    def __big_pop(self):
        sample_size = ((self.z * 2) * self.std * (1 - self.std)) / (self.e)**2
        return sample_size
    
    def __get_samplesize(self):
        sz = self.__big_pop() # sz : sample size
        if self.pop_size < self.treshold:
            sz = self.__short_pop()
        return int(sz)

class asyncScraper(object):
    def __init__(self, urls, useSample=False):
        self.urls = urls
        self.size = len(self.urls)
        self.master_dict = {}
        self.default_vid = {
                'Youtuber'       : None,
                'Type'           : None,
                'duration' : None
        }
        self.duration = 0
        # Run The Scraper:
        asyncio.run(self.__main())

    async def __fetch(self, session, url):
        try:
            async with session.get(url) as response:
                # 1. Extracting the Text:
                text = await response.text()
                # 2. Extracting the video info:
                video_info = await self.__extract_data(text, url)
                return url, video_info
        except Exception:
            return url, self.default_vid
            
    async def __extract_data(self, text, url):
        try:
            page = BeautifulSoup(text, 'html.parser')
            
            zone = page.find("div")
            
            content = lambda x: x["content"] if type(x) != type(None) else None
            
            time_converter = lambda x: round(isodate.parse_duration(x).total_seconds()/60,2) if type(x) != type(None) else None
            vid = {
                'Youtuber'       : content(zone.find('link', itemprop='name')),
                'Type'           : content(zone.find(itemprop='genre')),
                'duration' : time_converter(content(zone.find(itemprop='duration'))) 
            }
            return vid
        except Exception:
            return url, self.default_vid

    async def __main(self):
        tasks = []
        headers = {
            "user-agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"}
        start = time()
        async with aiohttp.ClientSession(headers=headers) as session:
            for url in self.urls:
                tasks.append(self.__fetch(session, url))

            results = await asyncio.gather(*tasks)
            
           # Store the result
            for result in results:
                if result is not None:
                    url = result[0]
                    self.master_dict[url] = result[1]
                else:
                    continue
        self.duration = time() - start
        
                    
    def get_dataframe(self):
        df = [self.master_dict[url] for url in self.urls]
        
        return pd.DataFrame(df).fillna(value=np.nan)