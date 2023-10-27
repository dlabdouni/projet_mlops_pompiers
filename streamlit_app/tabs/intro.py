import streamlit as st
from PIL import Image
import pandas as pd
import requests




sidebar_name = "Home"


def run():
    # Chargement de l'image
    image_path = "Image panier.png"
    image = Image.open(image_path)

    # Affichage de l'image
    st.image(image)

    # Affichage du titre en tant que superposition textuelle
    st.markdown("<h1 style='position: relative; top: 50%; left: 49%; transform: translate(-48%, -133%); color: white;'>London Fire Brigade Response Time</h1>",
                unsafe_allow_html=True)
    
    st.subheader('''Welcome to the London Fire Brigade Response Time page.''')
    
    st.write('''This project focuses on the topic of "London Fire Brigade Response Time." Its primary goal is to provide estimates for the response times between the call and the arrival of the London Fire Brigade at the scene of the incident. The London Fire Brigade is renowned as the United Kingdom's most active fire and rescue service and ranks among the largest firefighting and rescue organizations globally.''')
    st.write('''To achieve this goal, the project will have examined the factors that influence response times, such as geographic location, the timing of the intervention (hours, months), or the resources mobilized. For this purpose, it will have explored data sources related to incident records and mobilization records, which are accessible as open-source data on the London Fire Brigade website.
    
Here, after proper identification, you will enter multiple variables, while others remain invisible (such as the time of the call) because they are automated.This project is not yet fully completed; it is a work in progress. Ideally, the goal would be to transform and automate the variables in such a way that you would only need to provide two or three pieces of information (such as the address, for example) to obtain a response time prediction.
''')
            

            
