import streamlit as st
import requests


sidebar_name = "Predict Response Time"

def run():
    # Interface utilisateur Streamlit
    st.title("London Fire Brigade Response Time")


    # URL de l'API FastAPI
    api_url = "http://0.0.0.0:8001" 

    # Fonction pour effectuer l'authentification
    def authenticate(username, password):
        response = requests.post(f"{api_url}/bienvenue", auth=(username, password))
        return response

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # Bouton pour effectuer l'authentification
    if not st.session_state.authenticated:
        st.subheader('''Please enter your credentials.''')
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Authenticate"):
            if not username or not password:
                st.error("Veuillez fournir un nom d'utilisateur et un mot de passe.")
            else:
                response = authenticate(username, password)
                if response.status_code == 200:
                    st.session_state.auth_info = (username, password)
                    st.session_state.authenticated = True
                    st.success("Authentification réussie. " + response.json()['message'])
                else:
                    st.error("Échec de l'authentification. " + response.json()['detail'])

    if st.session_state.authenticated:
        
        # Widgets pour collecter les données d'entrée
        
        borough_code = st.text_input("Borough Code")
        ward_code = st.text_input("Ward Code")
        station_ground = st.text_input("IncidentStationGround")
        
            # Min_value et max_value pour contraindre l'entrée à des entiers
        easting = st.number_input("Easting", min_value=503550, max_value=560450, step=100)
        northing = st.number_input("Northing", min_value=155950, max_value=200850, step=100)
        num_stations_with_pumps = st.number_input("Num Stations With Pumps Attending", min_value=1, max_value=15, step=1)
        num_pumps_attending = st.number_input("Num Pumps Attending", min_value=1, max_value=15, step=1)
        pump_count = st.number_input("Pump Count", min_value=1, max_value=160, step=1)
        pump_hours_round_up = st.number_input("Pump Hours Round Up", min_value=1, max_value=700, step=1)
        pump_order = st.number_input("Pump Order", min_value=1, max_value=15, step=1)
        delay_code_id = st.number_input("Delay Code ID", min_value=1, max_value=15, step=1)

        # Bouton pour effectuer la prédiction
        if st.button("Predict"):
            auth_info = st.session_state.auth_info

        # Préparation des données d'entrée
            data = {
                "IncGeo_BoroughCode": borough_code,
                "IncGeo_WardCode": ward_code,
                "IncidentStationGround": station_ground,
                "Easting_rounded": easting,
                "Northing_rounded": northing,
                "NumStationsWithPumpsAttending": num_stations_with_pumps,
                "NumPumpsAttending": num_pumps_attending,
                "PumpCount": pump_count,
                "PumpHoursRoundUp": pump_hours_round_up,
                "PumpOrder": pump_order,
                "DelayCodeId": delay_code_id
            }


            try:
                response = requests.post(f"{api_url}/predict", json=data, auth=auth_info)

                if response.status_code == 200:
                    response_text = response.text
                    st.success(response_text)
                else:
                    response_data = response.json()
                    error_message = response_data.get("error", "Erreur")
                    st.error(f"{error_message}: {response_data.get('detail', response.text)}")
            except requests.exceptions.RequestException as e:
                st.error("Erreur lors de la requête à l'API : " + str(e))
