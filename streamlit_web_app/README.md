# Methane detection
This readme file provides instructions on setting up and running the web app for the methane detection tool.

## Installation
To install the necessary dependencies, follow the instructions below based on your operating system.

- For Windows

1. Create a virtual environment by running the following command in your command prompt or terminal:

```
cd /path to streamlit_web_app/
python -m venv QB_hackathon_env
```

2. Activate the virtual environment by running the following command:

```
QB_hackathon_env\Scripts\activate
```

3. Install the required packages using pip:

```
pip install numpy pandas matplotlib streamlit torch torchvision Pillow opencv-python folium streamlit_folium

```


- For macOS

1. Create a virtual environment by running the following command:

```
cd /path to streamlit_web_app/
python3 -m venv QB_hackathon_env
```


2. Activate the virtual environment by running the following command:

```
source QB_hackathon_env/bin/activate
```


3. Install the required packages using pip3:

```
pip install numpy pandas matplotlib streamlit torch torchvision Pillow opencv-python folium streamlit_folium
```

- Pretrained model

Put the model weight file inside the models subdirectory.
The file can be found here https://centralesupelec-my.sharepoint.com/:u:/g/personal/antoine_cloute_student-cs_fr/EQtmabBevsZJpJT3fLvzbY4B34iMsR-7kUZHIkRnqjF9SQ?e=Q9BPgn


## Running the Application
Once the installation is complete, you can run the application using the following command:

```
streamlit run app.py
```
