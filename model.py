# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 15:41:21 2021

@author: vwork
"""

import os

# General variables ----------------------------------------------------------

# Path to the project folder

app_info = {
    "name"   : "Youtube analyzer",
    "logo"   : "\U0001F52C",
    "layout" : "centered",
    "author" : {
        "last_name" : "Williamu",
        "first_name" : "Vaihau"
        }
    }

app_info["author"]["initials"] = app_info["author"]["last_name"].upper()[0] +\
                                 app_info["author"]["first_name"].upper()[0]
                                 
PROJECT_FOLDER = os.path.dirname(__file__)

pth = lambda fold : os.path.join(PROJECT_FOLDER, fold)


folder = {
    'image' : pth('Image'),
    'data' : pth('Data')
    }

path = {
        '1' : os.path.join(folder['data'], 
                           'preset_data_original_crypted.feather'), 
        
        'scrap' :  os.path.join(folder['data'], 
                                'preset_data_transformed_crypted.feather')
        }

url = {
       "linkedin" : {
           "img" : "https://img.shields.io/badge/Vaihau-0077B5?style=for-the-badge&logo=linkedin&logoColor=white&link=https://www.linkedin.com/in/vaihau-williamu/",
           "url" : "https://www.linkedin.com/in/vaihau-williamu/"
           } ,
       "github" : {
           "img" : "https://img.shields.io/badge/Vaihau-171B23?style=for-the-badge&logo=github&logoColor=white&link=https://github.com/WVaihau",
           "url" : "https://github.com/WVaihau"
           }
       }

page_config = {
    "page_title" : app_info["name"] + " - " + app_info["author"]["initials"],
    "layout" : app_info["layout"],
    "page_icon" : app_info["logo"]
    }

type_file = ['json']

# View variables -------------------------------------------------------------

mode_proposition = {
    'An existing dataset' : 'View what this web app can do with a default dataset. More functionality are available if you upload your own data !',
    'My own dataset' : 'Explore all your data with all the features YoutubeAnalyzer has to offer !'
    }

sidebar_section_logo = {
    "tuner" : "\U00002699",
    "network" : "\U0001F4CC",
    "upl" : "\U0001F4C1", # uploader
    "dl" : "&#x1f4e5" # download
    }

linkedin_btn = f"[![Connect]({url['linkedin']['img']})]({url['linkedin']['url']})"
github_btn = f"[![Connect]({url['github']['img']})]({url['github']['url']})"
network_btn = f"{linkedin_btn}&nbsp{github_btn}"

## My own dataset menu
mod_menu = {
    'Exploratory Data Analysis' : "Explore what your data tell about you.",
    'Video Explorer' : "Rediscover your history with our video explorer !"
    }

## Already existing dataset
preset_dataset = {
    "name" : "preset dataset",
    "path" : path["1"]
    }

file_in = None # uploader variable
error_file_in = False # state if there was an error while loading the dataset

# Controller variables -------------------------------------------------------
time_indicator = {
        'year'    : lambda dt: dt.year,
        'month'   : lambda dt: dt.month,
        'weekday' : lambda dt: dt.weekday(),
        'day'     : lambda dt: dt.day,
        'hour'    : lambda dt: dt.hour
}

col_to_drop = ['header', 'products', 'details', 'subtitles', 'description']

str_to_drop = ['Vous avez regardé ', 'Watched']



# Text filter
txt_filter = [
    r"[,\d\|\]\[\!\?\%\(\)\/\"]",
    r"\&\S*\s",
    r"\-"
    ]

alphabet = 'a z e r t y u i o p q s d f g h j k l m w x c v b n'.split()

taboo_word = ['the', 'to', 'a', 'in', 'or', 'and', 'your', 'you', 'of', 'i', 'for', 'my', 'yours', 'de'] + alphabet
