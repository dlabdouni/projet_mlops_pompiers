from collections import OrderedDict
import streamlit as st
from tabs import intro, pred_time, credits


st.set_page_config(
    page_title= "London Fire Brigade Response Time", page_icon="🔥")

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)



# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (pred_time.sidebar_name, pred_time),
        (credits.sidebar_name, credits),
    ]
)


def run():
      
    st.sidebar.image(
        "blason_LFB_1.png",
        width=250,
    )
    
                
    st.sidebar.markdown('<h1 style="color: #000000;">MENU</h1>', unsafe_allow_html=True)
    
    
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0, key="tab_selector")

    

    
      
  
    tab = TABS[tab_name]
    
    
    tab.run()


if __name__ == "__main__":
    run()
