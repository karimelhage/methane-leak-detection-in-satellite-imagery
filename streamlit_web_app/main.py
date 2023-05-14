# main.py
import streamlit as st
from pages.home import home_page

# Define the pages
PAGES = {
    "Home": home_page
}

# Use a selectbox for navigation
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Display the selected page with the session state
page = PAGES[selection]

# Run the function to display the selected page
page()