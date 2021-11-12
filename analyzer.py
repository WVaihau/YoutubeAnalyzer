# -*- coding: utf-8 -*-
"""
@author: WILLIAMU Vaihau
"""

import streamlit as st
import model
import controller as ctrl

# Configuration --------------------------------------------------------------
st.set_page_config(**model.page_config)

# Layout ---------------------------------------------------------------------

## Sidebar --

### Subheader
st.sidebar.title(model.app_info["logo"] + " " + model.app_info["name"])

mode = st.sidebar.selectbox('Work with', model.mode_proposition)

mode_description = st.sidebar.container()

view_mode = st.sidebar.container()

tuner = st.sidebar.container()

download = st.sidebar.container()

social_network = st.sidebar.container()

## Main --

content = st.container()

# Condition ------------------------------------------------------------------

## SIDEBAR

with mode_description: # Display mode description + params
    st.write(model.mode_proposition[mode])
    
    if mode == "My own dataset":
        # File uploader
        file_in = st.sidebar.file_uploader(model.sidebar_section_logo["upl"] +\
                                           " Upload your dataset", 
                                           type=['json', 'feather', 'csv'])
        
        model.file_in = file_in
        
        if file_in is not None: # If there is an uploaded file
            try:
                title = "Exploring " + mode.lower() + ' : ' + file_in.name
                file = ctrl.load_data(file_in) # load it
                model.error_file_in = False
            except:
                st.warning("A problem occured with your file please check it before uploading it in this web app")
                model.error_file_in = True
        else:
            model.error_file_in = True
            view_dropdown = None
    else:

        title = "Exploring the " + model.preset_dataset["name"]
        # Load the preset data
        file = ctrl.load_data(model.preset_dataset['path'], preset=True)
        model.error_file_in = False                 

    if model.error_file_in != True: # If there was no problem loading the dataset 
        years = [ str(y) for y in file['year'].unique().tolist()]
        subdf = ctrl.load_subdf(file)
        info = ctrl.get_insight(file)

with view_mode: # Display the menu for "My own dataset"
    if mode == "My own dataset" and model.file_in != None:
        view_dropdown = st.sidebar.selectbox("Mode", model.mod_menu.keys())
        st.sidebar.markdown(model.mod_menu[view_dropdown])


with tuner: # Display the correct available tuner
    if model.error_file_in != True:
        st.sidebar.subheader(model.sidebar_section_logo["tuner"] + ' Tuner')
        if mode == "My own dataset":
            if view_dropdown == 'Exploratory Data Analysis': # EDA Tuner
                analysis_range_container = st.sidebar.expander("Range of the analysis")
                with analysis_range_container:
                    st.markdown("""
                                As the confidence level increases, so does the accuracy. 
                                
                                The larger the margin of error, the smaller the sample size and the faster the results. 
                                
                                If you decide to take a confidence level of 100% depending on the size of your data the application may take time to generate the analysis.
                                """)
                                
                    confidence_level = st.selectbox('Confidence level', ('99%', '95%','90%', "100%"))
                    
                    margin_error = None
                    if confidence_level != "100%":
                        margin_error = st.selectbox('Margin error', ("5%", "1%"))
                
                sample_params = None
                if confidence_level != "100%":
                    sample_params = {
                        "confidence_level" : confidence_level,
                        "margin_error" : int(margin_error.split('%')[0]) * .01
                    }
            elif view_dropdown == 'Video Explorer': # VE tuner
                vid_explorer = st.sidebar.expander("Video selector")
                btn_random = False
                with vid_explorer:
                    col_v1, col_v2 = st.columns(2);
                    
                    with col_v1: # Select the year
                        opt_y = st.selectbox('Year', years)
                    
                    with col_v2: # Select which video
                        opt_type = st.selectbox('Type', ('First', 'Last', 'Random'))
                        
                    if opt_type == 'Random': # if random video
                        col_v11, col_v12, col_v13 = st.columns(3)
                        
                        with col_v11:
                            st.write('')
                            
                        with col_v12:
                            btn_random = st.button('Random', help="Select another random video")
                        
                        with col_v13:
                            st.write('')
                    
                    st.markdown("Note : **The video may not be displayed if it has been deleted or made private**")
        else:
            confidence_level = "100%"
            view_dropdown = 'Exploratory Data Analysis'
        
        if view_dropdown == 'Exploratory Data Analysis': # General EDA Tuner
            # Race chart tuner
            race_item = "Youtuber"
            race_time = "Year"
            
            race_chart_expender = st.sidebar.expander('Race chart')
            with race_chart_expender:
                if confidence_level == '100%':
                    cc1, cc2 = st.columns(2)
                    
                    with cc1:
                        race_item = st.selectbox('Item', ('Youtuber', 'Type'))
                    with cc2:
                        race_time = st.selectbox('Time indicator', ('Year', 'Weekday'))
                else:
                    st.markdown("""
                                In order to **plot the race chart** you must **select a confidence level of 100%** in the **range of the analysis**
                                """)
            # Ridge line tuner
            ridgeline_container = st.sidebar.expander('Ridgeline chart')
            with ridgeline_container:
                ridge_topic = st.selectbox('Subject', ('Youtuber', 'Type'))
                ridge_col = st.selectbox('Group by', ('Hour', 'Weekday'))
            
            if mode == 'My own dataset':
                # Analysis about the top tuner
                analysis_container = st.sidebar.expander('Analysis about the top')
                with analysis_container:
                    selectbox_genre = st.selectbox('Topic', ('Youtuber', 'Type'))
            
         
with social_network: # Display the social network button
    st.sidebar.subheader(model.sidebar_section_logo["network"] + ' Network')
    st.sidebar.write(model.network_btn)
    
## MAIN CONTENT
with content:
    if model.error_file_in == True:
        if mode == "My own dataset":
            ctrl.page_no_uploaded_file()
        elif mode == 'An existing dataset':
            ctrl.page_error_load_preset_data()
    else:    
        st.title(title)
        
        # General analysis
        
        ## Metrics
        col1, col2, col3 = st.columns(3)

        col1.metric('Number of Videos watched', str(info['nbr_watched']))
        col2.metric('From',str(info["period"]['from']))
        col3.metric('To', str(info["period"]['to']))
        
        st.write('')
        
        ## Graph
        if view_dropdown == 'Video Explorer': ### Video Explorer
            st.subheader('My youtube history')
            sub_file = ctrl.apply_tuner_year(file, opt_y)
            
            if opt_type != 'Random':
                vid_link = ctrl.get_insight(sub_file)[opt_type.lower()]
            else:
                vid_link = ctrl.random_vid(sub_file)
            
            if btn_random != False:
                vid_link = ctrl.random_vid(sub_file)
            
            st.video(vid_link)
        elif view_dropdown == 'Exploratory Data Analysis':
            st.subheader('General analysis')
            
            ### Line chart
            info_line, graph_line = ctrl.line_chart(file)
            
            ### Heat map
            heatmap_grph = ctrl.heatmap_by_weekday(file)
            st.plotly_chart(graph_line, use_container_width=True)
            st.markdown('Your highest number of videos watched per hour is in **{}** at the **{}th hour** of the day with over **{} videos watched**.'.format(info_line['year'], info_line['hour'], info_line['value']))
            st.plotly_chart(heatmap_grph, use_container_width=True)
            
            
            
        # Web Scraping Analysis
        
        ## Scrap data
        if mode == 'An existing dataset':
            df = ctrl.load_preset(model.path['scrap'])
        elif view_dropdown == 'Exploratory Data Analysis' and mode == 'My own dataset':
            st.subheader('EDA with the scraped data from the web')
            with st.spinner('Scrapping your data from the web. This might take a while...'):
                df = ctrl.fetch_data(file, sample=sample_params)
            
            ### Scrapped Metrics : My own Dataset - confidence level != 100% 
            if confidence_level != "100%":
                ccc1, ccc2, ccc3 = st.columns(3)
            
                ccc1.metric('Sample size', str(df.shape[0]), str(df.shape[0] - file.shape[0]))
                ccc2.metric('Confidence level',confidence_level)
                ccc3.metric('Margin Error', margin_error)
                st.info('The following information and graphs are based on the sample taken from the dataset')
        
            ## Download section
            csv = ctrl.convert_df(df) # Prepare the data for download

        ## Graph
        
        if view_dropdown == 'Exploratory Data Analysis':
            ### Race chart
            if confidence_level == "100%":
                race_chart = ctrl.race_chart(df, race_item, race_time.lower())
                st.write(race_chart)
                
            ### Bar chart
            cc1, cc2 = st.columns(2)
            
            with cc1:
                fig_type, info_type = ctrl.bar(df, 'Type', direction='h')
                st.plotly_chart(fig_type, use_container_width=True)
                st.markdown("""
                            The majority of the videos watched were in the **{}** category with over **{} videos watched**
                            """.format(info_type['Type'], info_type['count']))
            with cc2:
                fig_ytb, info_ytb = ctrl.bar(df, 'Youtuber')
                st.plotly_chart(fig_ytb, use_container_width=True)
                st.markdown("""
                            Here is the list of Youtubers that you have watched the most videos. At the first place we find **{}** with more than **{} videos watched**
                            """.format(info_ytb['Youtuber'], info_ytb['count']))
            
            ### Ridgeline 
            st.plotly_chart(ctrl.ridgeline(df, ridge_topic, ridge_col[0].lower()+ridge_col[1:]),use_container_width=True)
            
            
            if mode == 'My own dataset':
                ## Analysis about the top
                info = ctrl.vids_info(df, info_ytb, 'Youtuber')
                if selectbox_genre == "Type":
                    info = ctrl.vids_info(df, info_type, 'Type')
                ctrl.details(df, info, selectbox_genre)
                
                ### Instanciate download section
                with download:
                    st.sidebar.subheader(model.sidebar_section_logo["dl"] + \
                                         ' Download')
                    st.sidebar.markdown('Explore your data with your own method by downloading the file we used to generated the different analysis.')
                    st.sidebar.download_button(
                            label = "Download my data",
                            data  = csv,
                            file_name = 'watched_history_by_wv.csv',
                            mime = 'text/csv'
                        )