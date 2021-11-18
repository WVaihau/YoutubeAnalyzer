# Youtube Analyzer

Whether it's daily, weekly or monthly, we've all spent time on Youtube. However, we do not necessarily know the extent to which we consume this platform. Through Youtube Analyzer you will be able to explore what your Youtube data says about you. You will be able to rediscover the videos you have previously seen and much more. Youtube Analyzer is above all a tool to help you understand your use of YouTube.

If you are interested in analyzing your Youtube consumption [click here](https://share.streamlit.io/wvaihau/youtubeanalyzer/main/analyzer.py).

@Author : [WILLIAMU Vaihau](https://www.linkedin.com/in/vaihau-williamu/)

## 1. The technologies

This project is totally realized in python 3.9.6.

Main modules used :

ETL

- [Aiohttp](https://pypi.org/project/aiohttp/) (Web Scraping)
- [Pandas](https://pypi.org/project/pandas/)
- [Numpy](https://pypi.org/project/numpy/)
- [feather-format](https://pypi.org/project/feather-format/)

Data Encryption

- [Pyarrow](https://pypi.org/project/pyarrow/)
- [Cryptography](https://pypi.org/project/cryptography/)

Exploratory Data Analysis

- [Pandas](https://pypi.org/project/pandas/)
- [Plotly](https://pypi.org/project/plotly/)

Web App

- [Streamlit](https://pypi.org/project/streamlit/)

## 2. What it can do for you

The sibar serves as a navigation menu. All or at least most of the interactions you will have with this application can be found here.

Here is an outline of what the application can do for you:

- Work with an existing dataset

This view represents the default view of the web application. It shows you some graphics that the application can generate with a youtube history.

Remark : You will have access to much more functionality if you import your own data.

- Work with your own dataset

When you select this option you can import your own Youtube history.

Once your data is imported you will be presented with two options :

1. Exploratory Data Analysis

    This view allows you to see what your history can say about you with some insight. You can customize your experience with the Tuner in the sidebar.

2. Video Explorer

    This view allows you to explore your youtube history by year. You have the possibility to see the first or the last video viewed in a given year.

## 3. Note

- Web Scraping

The default history we receive from youtube allows us to retrieve the urls of the watched videos. So I wrote an Asynchronous Web Scraping class that allows to get additional information about a given video from the youtube history.

- Exploratory Data Analysis

In spite of the time saving that the Asynchronous Web Scraping class allows to have, the complexity of the program will take more time if the history is consequent. So I applied some sampling principle to represent the whole Youtube history.

**The best way to explore the application is to use it : [click here](https://share.streamlit.io/wvaihau/youtubeanalyzer/main/analyzer.py)**

**[END DOCUMENTATION]**