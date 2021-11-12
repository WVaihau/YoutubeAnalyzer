# -*- coding: utf-8 -*-
"""
@author: WILLIAMU Vaihau
"""
#VIEW
import streamlit as st

#GRAPH
import plotly.graph_objects as go
import plotly.express as px
from raceplotly.plots import barplot
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#DATAFRAME
import pandas as pd
import feather
import model
from random import randint as rd
from PIL import Image
import re
import numpy as np
from cryptography.fernet import Fernet
from pyarrow import BufferReader

#WEB SCRAPING
from scraper import asyncScraper, SampleSize 

# Private --------------------------------------------------------------------
def __to_datetime_type(df:pd.DataFrame, cols:list)->pd.DataFrame:
    """
    
        Convert the type of each column in cols to datetime 
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the columns to convert 
    cols : list
        Column to convert

    Returns
    -------
    df : pd.DataFrame
        Former dataframe with the converted colummns

    """
    for col in cols:
        df[col] = df[col].apply(pd.to_datetime)
    
    return df

def __add_time_indicator(df:pd.DataFrame, colref:str, indicators:list, suffix=''):
    """
        Add a list of time indicators in the df dataframe based on the
        colref column with the possibility to add a suffix for each time 
        indicator column name.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to add the different time indicator.
        
    colref : str
        The time column from which we will retrieve the different time data. 
    indicators : list
        List of indicators to be retrieved.
    suffix : TYPE, optional
        Text to be added at the end of the column names of the time indicators. 
        The default is ''.

    Returns
    -------
    df : pd.DataFrame
        The dataframe containing the different time indicator

    """
    
    # Collects all available time indicators
    type_indicator = model.time_indicator
    
    #For each of the desired time indicators
    for indicator in indicators:
        if indicator in type_indicator: # If it is a valid indicator
            
            # The colname will be the indicator
            col_name = indicator
            
            # Add the suffix if there is one
            if suffix != '':
                col_name += '_' + suffix
            
            # Add the column
            df[col_name] = df[colref].map(type_indicator[indicator])
    
    return df

def __clean_prefix(df:pd.DataFrame, col:str, to_del:dict)->pd.DataFrame:
    """
    Delete the prefix to_del in the column col in the dataframe df.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the col to clean.
    col : str
        Name of the column where to clean the prefix
    to_del : dict
        Dictionnary containing all the string to get rid of.

    Returns
    -------
    df : pd.DataFrame
        Dataframe containing the col cleaned.

    """
    for key, taboo_word in to_del.items():
        df[key] = __drop_redundancy(df[key], col, taboo_word) 
    
    return df

def __drop_redundancy(df:pd.DataFrame, col:str, txt:str)-> pd.DataFrame:
    """
    Removes text repetitions txt in a specific column col of a dataframe df.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the col to handle.
    col : str
        Name of the column to apply the transformation.
    txt : str
        Text to drop.

    Returns
    -------
    df : pd.DataFrame
        Dataframe containing the cleaned col.

    """
    df[col] = pd.Series([ ''.join(elt.split(txt)) for elt in df[col].tolist() ])
    return df

def __interval(liste, size=10000):
    size_block = int(len(liste) / size)
    
    reste = liste[size * size_block:]
    
    intervals = [liste[size * (i - 1):size * i] for i in range(1, size_block + 1)] 
    
    if len(reste) >= 1:
        intervals += [reste]
    
    return intervals

def __text_filter(text:str):
    """
        Clean a text with specific filter.

    Parameters
    ----------
    text : str
        Text to clean.

    Returns
    -------
    text : str
        Cleaned text.

    """
    
    text = text.lower() # mettre les mots en minuscule
    
    # Apply each filter on the text
    for filtre in model.txt_filter:
        text = re.sub(filtre, '', text)
    
    return text

def __transform(df:pd.DataFrame)->pd.DataFrame:
    """
    Apply transformation on a dataframe

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to transform.
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame transformed.
    """
    # Retrieves the data to be applied
    str_to_drop = model.str_to_drop # list of text to delete
    col_to_drop = model.col_to_drop # list of column to drop
    
    
    # Convert to datetime
    df = __to_datetime_type(df, ['time'])
    
    # Adds time indicators
    df = __add_time_indicator(df, 'time', ['year', 'weekday', 'hour'])
    
    # Delete the designated text
    for col in str_to_drop:
        df = __drop_redundancy(df, 'title', col)
    
    # Delete the designated columns if they exist
    for col in col_to_drop:
        try:
            df = df.drop(col, axis=1)
        except:
            pass
    return df

def __sizeby(df, col, nbr=5, ascend = False):
    """
    Count for specific col in a dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the col to count from.
    col : str
        Column name to count from.
    nbr : int, optional
        Number of value to retrieves. The default is 5.
    ascend : boolean, optional
        Sort order by ascending. The default is False.

    Returns
    -------
    sub_df : pd.DataFrame

    """
    sub_df = df.groupby(col).size().sort_values(ascending=ascend).head(nbr).reset_index()
    sub_df.rename(columns={0:"count"}, inplace=True)
    
    return sub_df

def __feather_file_decryptor(file:str, key:bytes):
    """
    Decypher the crypted feather file into an available dataframe

    Parameters
    ----------
    file : str
        Crypted file
    key : bytes

    Returns
    -------
    df : DataFrame
        Decrypted file
    """
    
    # load decryptor
    fernet = Fernet(key) 
    
    # Get the crypted file
    with open(file, "rb") as file:
        encrypted_byte_file = file.read()
    
    # Decrypt the file in bytes
    decrypted_byte_file = fernet.decrypt(encrypted_byte_file)
    
    # file in bytes to dataframe
    df = feather.read_dataframe(BufferReader(decrypted_byte_file))
    
    return df

# Public ---------------------------------------------------------------------

@st.cache
def fetch_data(df_in, sample=None, max_scraper_size=10000, col='titleUrl'):
    """
    Retrieves from a specified column the various data desired using a web 
    scraping method.

    Parameters
    ----------
    df_in : pd.DataFrame
        DESCRIPTION.
    sample : dict, optional
        Sample parameters. The default is None.
    max_scraper_size : int, optional
        Maximum capacity that the scraper can receive before being blocked. 
        The default is 10000.
    col : str, optional
        Name of the column that contains the urls to retrieve. 
        The default is 'titleUrl'.

    Returns
    -------
    final_df : pd.DataFrame
        Original dataframe containing in addition the set of columns retrieved
        from the scraper
    """

    size = df_in.shape[0] # By default take all the dataset size
    
    if sample != None: # If there are sample parameters take the calculated sample size
        size = SampleSize(size, **sample).samplesize
    
    # Select random individuals from the dataset, sort them by index and finaly reset the index
    df = df_in.sample(size).sort_index().reset_index().drop(columns=['index'])
    
    # Get the col which contains the urls
    video_urls = df[col].tolist()

    # Scraping ---------------------------------------------------------------
    
    # If the number of url is higher than the maximum quantity that the scraper can take in one time
    if len(video_urls) > max_scraper_size:
        # Distribute the urls in list of size lower or equal to the maximum size of the scraper
        urls_list = __interval(video_urls, size=max_scraper_size)
        
        # Retrieves the desired data for each list using the scraper
        data_by_list = {i: asyncScraper(urls_list[i]) for i in range(len(urls_list))}
        
        # Join the new data in one dataframe
        df_new_data = pd.concat([x.get_dataframe() for x in data_by_list.values()]).reset_index().drop(columns=['index'])
    else:
        # Retrieves directly the desired data using the scraper
        data = asyncScraper(video_urls)
        
        # Get the data dataframe
        df_new_data = data.get_dataframe()
    
    # Join results with the former dataframe ---------------------------------
        
    final_df = pd.concat([df, df_new_data], axis = 1)   
    
    # Return the final df
    return final_df

@st.cache
def convert_df(df): # Get the csv version of a dataframe
    return df.to_csv(index=False).encode('utf8')

@st.cache
def get_img(path): # Return an image
    image = Image.open(path)
    return image 

@st.cache
def apply_tuner_year(df, years): # Apply a tuner parameter
    d = df.query('year in ({})'.format(years))
    return d

def load_preset(file:str):
    """
    Load preset data

    Parameters
    ----------
    file : str
        preset dataset path

    Returns
    -------
    df : Dataframe
        DESCRIPTION.

    """
    
    #Get the decypher key
    key = st.secrets['key']
    
    df = __feather_file_decryptor(file, 
                       key)
    
    return df

def load_data(path, trf_func=__transform, preset = False):
    """
    Extract and Load a dataframe from path and apply if specified a transform
    function
    
    Parameters
    ----------
    path : UploadFile or path to a file
        File or path to generate a dataframe
        
    trf_func : function to apply, optional
        Contains all the transformation to apply on the dataframe. The default 
        is transform.
        
    preset : Boolean, optional
        Allow the function to know if we want to load the preset dataset

    Raises
    ------
    TypeError
        When a type of file is not managed by the function : csv, json, feather

    Returns
    -------
    df : Dataframe
        A youtube-history dataframe.

    """

    name = path

    verif_extension = lambda typ: typ == name.split('.')[-1]
    
    #Extract and Load the dataset
    if type(path) != str:
        name = path.name
    
    if preset == True: # Load preset data
        df = load_preset(path)        
    elif verif_extension('csv'):
        df = pd.read_csv(path)
    elif verif_extension('json'):
        df = pd.read_json(path)
    elif verif_extension('feather'):
        df = feather.read_dataframe(path)
    else:
        raise TypeError("Unknown type. '%s' is neither a csv, json or feather file " % path)
    
    if trf_func != None:
        df = trf_func(df)
    
    return df
    

def load_subdf(df):
    """
    
        Load all the sub dataframe needed

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    subdf : DataFrame

    """
    subdf = {}

    subdf['watch_hour_year'] = get_freq_by(df, ['year','hour'])
    
    subdf['watch_weekday_hour'] = df.groupby(['weekday','hour']).size().unstack()
    
    return subdf

def get_freq_by(df, grpby):
    """
    Get the frequence when grouped by grpby in df

    Parameters
    ----------
    df : Dataframe
        Dataframe to get freq
        
    grpby : 
        columns to group by.

    Returns
    -------
    DataFrame
        Dataframe in long format

    """
    k = df.groupby(grpby).size().unstack().reset_index()
    
    long = pd.melt(k, id_vars=grpby[0], value_vars=k.columns[1:])
    
    long.rename(columns={'value':'count'})
    
    return long

def get_insight(df):
    """
    Get insight from a youtube_history dataframe transformed by transform()

    Parameters
    ----------
    df : DataFrame
        A youtube_history dataframe transformed by transform()

    Returns
    -------
    df_info : dictionary
        Contains informations about a specific youtube_history dataframe transformed by transform()

    """
    df_info = {}
    df_info['period'] = {
    "from" : min(df['year']),
    "to" : max(df['year'])
    }
    df_info['nbr_watched'] = df.shape[0]
    df_info['first'] = df.iloc[-1, 1]
    df_info['last'] = df.iloc[0, 1]
    return df_info

def random_vid(df):
    """
    Get a random video

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing all the data

    Returns
    -------
    str
        URL of the random video

    """
    return df.iloc[rd(0,len(df)-1), 1]

def vids_info(df, maxi, col):
    """
    Get information about a video inside a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
    maxi : dict
        dictionnary containing the default informations of a dataframe.
    col : str
        Column to sort by.

    Returns
    -------
    maxi : dict
        Former dict with more information.

    """
    # sort by the select column
    d = df[df[col] == maxi[col]]
    
    # Url of the favorite videos
    url_fav = d.groupby('titleUrl').size().sort_values(ascending=False)

    # Retrieves the last and first video watched
    dico = {"last" : 0, "first" : -1}
    for key, val in dico.items():
        maxi[key + '_video'] = {
                "url" : d['titleUrl'].iloc[val],
                "time": d['time'].iloc[val],
            }
        maxi[key + '_video']['count'] = d[d['titleUrl'] == maxi[key + '_video']['url']].shape[0]

    # Add the information about the favorite video
    maxi['favorite_video'] = {
            "url" : url_fav.index[0],
            "count" : url_fav.values[0]
        }
    
    # Add the time between the first and last video watched
    maxi['period'] = pd.to_datetime(maxi['last_video']["time"]) - pd.to_datetime(maxi['first_video']["time"])
    
    # Add the wordcloud based on the column selected
    maxi['wordcloud'] = __text_filter(' '.join(d.title.values.tolist()))
    
    return maxi

def vids_line(title, info, metrics_value=None):
    """
    Display a video and its information on a line

    Parameters
    ----------
    title : str
        Title to display.
    info : dict
        All information about the video.
    metrics_value : dict, optional
        Information to display. The default is None.
    """
    
    # Write the title
    st.write(title)
    
    
    if metrics_value == None:
        metrics_value = {
        "Date" : info['time'].date(),
        "Number of time watched" : info['count']
        }
    nbrcol = 1 + len(metrics_value)
    cols = st.columns(nbrcol)
    
    cols[0].video(info['url'])
    
    i = 1
    for key in metrics_value.keys():
        cols[i].metric(key, str(metrics_value[key]))
        i += 1

def frequent_word(txt, seuil=10, nbr=4):
    """
    Get the frequent words inside txt for the research on youtube

    Parameters
    ----------
    txt : str
        Text where we must take the words.
    seuil : int, optional
        Minimum occurence. The default is 10.
    nbr : int, optional
        Number of word to get back. The default is 4.

    Returns
    -------
    str
        string of the most frequent words

    """
    # Word to delete
    taboo_word = model.taboo_word
    
    l = {}
    for elt in txt.split(): # For each word
        if elt in l: # if it exist increase its counter
            l[elt] += 1
        else: # else initialized it to 1
            l[elt] = 1
            
    # Filter with the seuil
    l1 = {i:l[i] for i in l if l[i]>=seuil}
    
    lw = []
    # Get the most frequent ones
    for key in {k: v for k, v in list(reversed(sorted(l1.items(), key=lambda item: item[1]))) if k not in taboo_word}.keys():
        if nbr > 0:
            nbr -= 1
            lw.append(key)
        
    return '+'.join(lw)

def details(df, info, col):
    """
    Display the details for a specific col

    Parameters
    ----------
    df : pd.DataFrame
    info : dict
    col : str
    
    """
    
    # Display a subheader
    st.subheader("Analysis about the top {} : {}".format(col, info[col]))
    
    # Display the metrics
    cm1, cm2 = st.columns(2)
    cm1.metric('Total number of videos watched', str(info['count']))
    cm2.metric('Time between the first and last video watched', str(info['period'].days) + ' days')
    
    # Display the first video watched and its metrics
    vids_line("First video watched", info['first_video'])
    
    # Display the last video watched and its metrics
    vids_line("Last video watched", info['last_video'])
    
    # Instanciate the favorite video metrics
    fav_metrics = {
        "count" : info["favorite_video"]['count']
        }
    
    # Display the favorite video and its metrics
    vids_line("Favorite video watched", info['favorite_video'], fav_metrics)
    
    # Display a subheader
    st.subheader('The most used words in the video titles of this {} :'.format(col[0].lower()+col[1:]))
    
    # Create and generate a word cloud image:
    wordcloud = WordCloud(width=900, height=500).generate(info['wordcloud'])
    
    # Display the generated image:    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
    
    # Instanciate the recommandation url
    recommandation_url = "https://www.youtube.com/results?search_query=" + frequent_word(info['wordcloud'], nbr=5)
    
    # Write it on screen
    st.markdown("""
                You can find on [this link]({}) a set of videos chosen from the most used words
                """.format(recommandation_url))

## Graphic function ----------------------------------------------------------

def bar(df, col, range_value=5, direction='v'):
    """
    Generate a plotly Bar graph based on a column of a dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the column to managed.
    col : str
        Column to managed.
    range_value : int, optional
        Number of individual to get. The default is 5.
    direction : str, optional
        Direction of the plot. The default is 'v'.
            v : vertical
            h : horizental

    Returns
    -------
    fig : plotly.graph_objects
        Graph bar.
    maxi : dict
        max value details
    """
    # Get the data
    ytb = df.groupby(col).size().sort_values(ascending = False).head(range_value).sort_values(ascending= direction!='v')
    
    fig = go.Figure() # Create a figure
    
    x_val, y_val = ytb.index, ytb.values # set the values
    
    ytick_label = False # Do not show the y-label

    maxi = {
        col : ytb.index[0], # Column to handle
        "count" : ytb.values[0] # Count of the column to handle
    }
    
    # Apply some modification if it is horizental mode
    if direction == 'h':
        x_val, y_val =  ytb.values, ytb.index # switch axis
        ytick_label = True # show the y-label
        # Add information about the max
        maxi[col], maxi['count'] = ytb.index[-1], ytb.values[-1]
    
    # Add the bar graph
    fig.add_trace(
        go.Bar(
            x = x_val,
            y = y_val,
            orientation = direction
        )
    )
    
    # Update the figure 
    fig.update_layout(
        title_text="Top {} {}".format(range_value, col),
        yaxis=dict(
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=ytick_label
        ),
        plot_bgcolor='#F5F5F5'
    )

    return fig, maxi


def line_chart(df):
    """
    Display the line chart

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    info_max : dict
        Information about the maximal value.
    fig :  plotly.graph_objects 
        the figure to display
    """
    # Group by year and hour
    grpby_year_hour = get_freq_by(df, ['year', 'hour']) # Group by year and hour
    
    maxi = lambda field: grpby_year_hour[grpby_year_hour['value'] == grpby_year_hour['value'].max()][field].values[0]
    
    # Get the fields
    fields =  grpby_year_hour.columns.values.tolist()

    # Instanciate the dictionnary which will contain informations about the max
    info_max = {field:maxi(field) for field in fields}

    # Create a figure
    fig = go.Figure(
        layout = go.Layout(
            title=go.layout.Title(text="Number of video watched through years by hour")
        )
    )
    
    # Display each line
    for year in grpby_year_hour['year'].unique():
        md = 'lines'
        line_size = 1
        if year == info_max['year']:
            md += '+markers'
            line_size = 4
    
        specific_df = grpby_year_hour.query(f'year =={year}')
        bar = go.Scatter(x           = specific_df['hour'], 
                         y           = specific_df['value'], 
                         mode        = md, 
                         name        = str(year),
                         line=dict(
                             width = line_size 
                         ),
                         connectgaps = True)
    
        fig.add_trace(bar)
    
    
    fig.add_annotation(x=info_max['hour'], 
                       y=info_max['value'],
                       text="{} videos watched".format(info_max['value']),
                       showarrow=True,
                       yshift=10,
                       font=dict(
                           family = "Arial",
                           size = 16,
                           color='#DC143C'
                       )
                      )
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            linewidth=2,
            linecolor='rgb(204, 204, 204)',
            ticks='outside'
        ),
        yaxis=dict(
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False
        ),
        autosize=True,
        plot_bgcolor='#F5F5F5',
        showlegend=True
    )

    
    return info_max, fig

def heatmap_by_weekday(d1):
    """
    Return a heatmap of d1 by weekday and hour. It must contains the column :
        - weekday
        - hour

    Parameters
    ----------
    d1 : pd.DataFrame

    Returns
    -------
    heatmap_weekday_hour : plotly.graph_objects

    """
    df = d1.groupby(['weekday','hour']).size().unstack().fillna(0)
    
    heatmap_weekday_hour = px.imshow(df,
                                     title = 'Frequency by weekday and hour')
    heatmap_weekday_hour.update_layout(
    yaxis = dict(
        tickmode = 'array',
        tickvals = d1.groupby(['weekday','hour']).size().reset_index()['weekday'].unique(),
        ticktext = 'Mon Tue Wed Thu Fri Sa Sun'.split()
    )
    )
    return heatmap_weekday_hour

def race_chart(df_in, itemcol, timecol):
    """
        Get a race chart based on the itemcol by the timecol.

    Parameters
    ----------
    df_in : pd.DataFrame
        Dataframe containing the itemcol and timecol.
    itemcol : str
        Item column name.
    timecol : str
        Time column name.

    Returns
    -------
    graph : plotly.graph_objects
    """
    # Get the count
    df = df_in.groupby([itemcol,timecol]).size().to_frame().sort_values([0], ascending = False).reset_index()
    df.rename(columns={0:'count'}, inplace=True)
    df['count'] = df['count'].map(lambda x: int(x))
    
    # Instanciate the raceplot
    raceplot = barplot(df,  item_column=itemcol, value_column='count', time_column=timecol)
    
    # Instanciate the graph
    graph = raceplot.plot(item_label = 'Top 10 ' + itemcol,
                 value_label = 'Number of video watched',
                 time_label = timecol.upper() + ' : ', ## overwrites default `Date: `
                 frame_duration = 900)
    
    # Modify the graph layout
    graph.update_layout(
        width=900,
        height=700,
        title='Race chart of {} through {}'.format(itemcol, timecol)
        )
    return graph

def ridgeline(df, topic, col, offset_limit=.5, offset_title=.5):
    """
        Get a ridgeline based on a topic and a col

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the information to generate the ridgeline chart
    topic : str
        Column name of the topic
    col : str
        Column name by which we will sort the topic
    offset_limit : float, optional
        Graph detail. The default is .5.
    offset_title : float, optional
        Graph detail. The default is .5.

    Returns
    -------
    fig : plotly.graph_objects

    """
    # Get data
    subject_list = df.groupby([topic]).size().sort_values(ascending=False).head(5).index
    
    # Filter
    dt = df[df[topic].isin(subject_list)]
    
    dt = dt.groupby([topic, col]).agg({col : 'count'}).rename(columns={col : 'count'}).reset_index()
    
    # Prepare graph data
    array_dict = {}
    for subject in subject_list:
        array_dict[f'x_{subject}'] = dt[dt[topic] == subject][col]
        array_dict[f'y_{subject}'] = dt[dt[topic] == subject]['count']
        array_dict[f'y_{subject}'] = (array_dict[f'y_{subject}'] - array_dict[f'y_{subject}'].min()) \
                                    / (array_dict[f'y_{subject}'].max() - array_dict[f'y_{subject}'].min())
    # Create the graph
    fig = go.Figure()
    for index, subject in enumerate(subject_list):
        fig.add_trace(go.Scatter(
                                x=[dt[col].min(), dt[col].max()+offset_limit], y=np.full(2, len(subject_list)-index),
                                mode='lines',
                                line_color='white'))
        fig.add_trace(go.Scatter(
                                x=array_dict[f'x_{subject}'],
                                y=array_dict[f'y_{subject}'] + (len(subject_list)-index) + .5,
                                fill='tonexty',
                                name=f'{subject}'))
        fig.add_annotation(
                            x=dt[col].min()-offset_title,
                            y=len(subject_list)-index,
                            text=f'{subject}',
                            showarrow=False,
                            yshift=10)
    fig.update_layout(
                    title='Top {} by {}'.format(topic[0].lower()+topic[1:], col),
                    showlegend=False,
                    xaxis=dict(title=col[0].upper()+col[1:]),
                    yaxis=dict(
                        showline=False,
                        zeroline=False,
                        showgrid=False,
                        showticklabels=False
                    )
                    )
    return fig

## Page procedure -------------------------------------------------------------
def page_no_uploaded_file():
    st.warning("Please upload your dataset on the sidebar")
    st.subheader("How to download your dataset")
    st.markdown(""" 
                1. Go to [Google Take Out](https://takeout.google.com/?hl=en) and connect to your google account
                2. Deselect All                    
                3. Go to the downpage and select Youtube and Youtube Music
                4. Click on Multiple formats and choose json for History
                5. Click Next Step
                6. On the next page click Create a report (It may take some time before your data is available)
                7. When your data are ready, download and unzip them
                8. Send on the web app uploader the file takout-number/Takeout/Youtube et Youtube Music/historique/watch-history.json
                """)

def page_error_load_preset_data():
    st.warning('An issue ocured with the dataset')