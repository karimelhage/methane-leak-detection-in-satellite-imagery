import streamlit as st
#from pages.detection import detection
#from pages.home import home
#from pages.impact_and_use_cases import impact_and_use_cases
#from pages.about_and_contact import about_and_contact

import torch
from torchvision import transforms
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from datetime import datetime

#PAGES = {
#    "Home": home,
#    "Detection": detection,
#    "Impact and Use Cases": impact_and_use_cases,
#    "About Us / Contact Us": about_and_contact
#}

# load plume locations data
df_plume = pd.read_csv('location_latlon.csv')

def plot_on_map(df):
    # create a map centered at the mean of the latitude and longitude
    map_center = [df['lat'].mean(), df['lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=2)

    # create a marker cluster layer
    marker_cluster = MarkerCluster().add_to(m)

    # add markers for each data point
    for index, row in df.iterrows():
        lat = row['lat']
        lon = row['lon']
        plume = row['plume']
        path = row['path']

        folium.Marker([lat, lon], popup=f"Plume: {plume}, Path: {path}").add_to(marker_cluster)

    return m

def home():
    st.title("Welcome to CleanR's Methane Emission Detection Tool")
    st.subheader("Using AI to Reduce Methane Emissions")

    st.write("""
    ## About CleanR
    CleanR is a fast-growing start-up specialized in Methane emissions reporting. We're on a mission to reduce methane emissions by providing a clear method for MRV: monitoring, reporting, and verification. Utilizing the power of satellite imagery and deep learning, we're able to detect potential methane leaks, helping us take a step forward in environmental preservation.

    ## The Problem
    Methane is one of the most potent greenhouse gases, and its leaks pose a significant challenge in the fight against climate change. Early detection of methane leaks can help to significantly reduce the environmental impact.

    ## Our Solution
    Our tool uses a deep learning model trained on satellite images to detect potential methane plumes. Once a grayscale satellite image is uploaded, our model analyzes it and highlights areas with potential methane leaks, providing a confidence score along with the prediction. It's an effective way to monitor large areas and identify potential methane leaks quickly.

    ## How It Works
    1. **Upload a grayscale satellite image in .tif format:** Our tool accepts grayscale satellite images in .tif format for analysis.
    2. **Our model analyzes the image:** Using deep learning, our model identifies potential methane leaks in the image.
    3. **View the results:** Our tool highlights areas of potential methane leaks and provides a confidence score for its prediction.

    Ready to start? Navigate to the 'Detection' page on the sidebar.
    """)

    st.header("Contact Us")
    st.markdown("For any inquiries or feedback, please contact us at: info@cleanr.com")

    # User input for filtering
    plume_filter = st.selectbox('Choose plume filter', options = ['Both', 'Plume', 'No Plume'], index = 0)

    # Filter data based on user selection
    if plume_filter == 'Plume':
        df_filtered = df_plume[df_plume['plume'] == 'yes']
    elif plume_filter == 'No Plume':
        df_filtered = df_plume[df_plume['plume'] == 'no']
    else:
        df_filtered = df_plume

    # Call your function with the filtered data
    my_map = plot_on_map(df_filtered) #plot_on_map(df_filtered, plume_filter)

    folium_static(my_map)

def impact_and_use_cases():
    st.title("Impact and Use Cases")

    st.markdown("""
    ## Impact
    Methane is one of the most potent greenhouse gases, over 25 times more potent than carbon dioxide in terms of heat-trapping capability. By detecting methane leaks early, we can significantly reduce the environmental impact and contribute to the fight against climate change.

    Our tool can help various industries and sectors, including oil and gas, waste management, and agriculture, by identifying and quantifying methane emissions, leading to more sustainable operations.

    ## Use Cases
    ### Oil and Gas Industry
    Methane is a major component of natural gas. Unintended leaks during extraction, storage, and transportation can contribute significantly to greenhouse gas emissions. Our tool can help detect leaks early and prevent environmental damage.

    ### Waste Management
    Landfills are a major source of methane emissions. By using our tool, waste management facilities can monitor their sites for methane leaks and take appropriate action.

    ### Agriculture
    Methane is produced by certain agricultural practices, particularly those involving livestock. Our tool can help farmers identify potential sources of methane emissions on their farms and develop more sustainable practices.

    ## Testimonials
    > "CleanR's Methane Emission Detection Tool has been instrumental in helping us identify and fix several leaks in our natural gas infrastructure. It's easy to use and has already saved us a lot in potential fines and environmental damage." - **John Doe, Oil and Gas Company**

    > "As a landfill operator, being able to monitor our site for methane emissions has been incredibly helpful. We've been able to identify several problem areas and address them quickly." - **Jane Doe, Waste Management Company**
    """)

def create_model():
    # Load the pre-trained ResNet50 model
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    resnet50 = models.wide_resnet50_2(weights='Wide_ResNet50_2_Weights.IMAGENET1K_V2')

    # Modify the first layer to accept single-channel grayscale images
    resnet50.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Modify the last fully connected layer for binary classification with softmax activation
    num_classes = 2  # 2 classes: 1 or 0
    resnet50.fc = torch.nn.Sequential(
        torch.nn.Linear(resnet50.fc.in_features, num_classes)
    )

    return resnet50

# Define the model architecture
model = create_model()

# Load the state dictionary
model.load_state_dict(torch.load('models/resnet50_0.9268.pth', map_location=torch.device('cpu')))

# Set the model to eval mode
model.eval()

# Create a function to load and preprocess the image
def load_and_prep_image(image):
    # Open the image file
    image = Image.open(image)
    
    # Convert the image to float
    image = image.convert('F')

    # Define the transformation - convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor()])

    # Apply transformation and add an extra dimension for batch
    image = transform(image).unsqueeze(0)
    
    return image

# Grad-CAM function
def Grad_CAM(input_img, model):
    activation = []
    grad = []

    def activation_hook(module, input, output):
        activation.append(output)

    def gradient_hook(module, grad_input, grad_output):
        grad.append(grad_output[0])

    handle_activation = model.layer4[2].conv3.register_forward_hook(activation_hook)
    handle_gradient = model.layer4[2].conv3.register_backward_hook(gradient_hook)

    # Forward pass to get activations
    out = model(input_img)

    # Get the score for the target class (assuming index 1 is the positive class)
    score = out[0, 1]

    # Clear the gradients
    model.zero_grad()

    # Backward pass to get gradients
    score.backward()

    # Remove the hooks after collecting activations and gradients
    handle_activation.remove()
    handle_gradient.remove()

    # Get the gradients and activations collected in the hooks
    grads = grad[0].detach().numpy().squeeze()
    fmap = activation[0].detach().numpy().squeeze()

    # Expand the dimensions of fmap
    fmap = np.expand_dims(fmap, axis=0) if len(fmap.shape) == 2 else fmap

    # Calculate the Grad-CAM
    weights = np.mean(grads, axis=(1, 2))
    weights = weights.reshape(-1, 1, 1)
    weights = np.repeat(weights, fmap.shape[1], axis=1)
    weights = np.repeat(weights, fmap.shape[2], axis=2)

    cam = np.sum(weights * fmap, axis=0)
    cam = np.maximum(cam, 0)

    # Resize the gradients and feature map to match the input image size
    resized_grads = cv2.resize(grads, (input_img.shape[3], input_img.shape[2]))
    resized_fmap = cv2.resize(fmap, (input_img.shape[3], input_img.shape[2]))

    # Calculate the Grad-CAM
    cam = np.sum(resized_grads * resized_fmap, axis=0)
    cam = np.maximum(cam, 0)

    # Normalize the Grad-CAM
    cam = cam / (np.max(cam) + 1e-10)

    # Apply color mapping to the Grad-CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Convert the input image to RGB
    npic = np.array(input_img.squeeze(), dtype=np.uint8)
    npic_rgb = cv2.cvtColor(npic, cv2.COLOR_GRAY2RGB)

    # Resize the heatmap to match the input image size
    resized_heatmap = cv2.resize(heatmap, (input_img.shape[3], input_img.shape[2]))

    # Overlay the heatmap on the input image
    cam_img = cv2.addWeighted(npic_rgb, 0.1, resized_heatmap, 0.5, 0)

    return cam_img


def detection():
    st.title("Methane Emission Detection")

    st.markdown("""
    ## Instructions
    1. Upload a grayscale satellite image in .tif format.
    2. Click on 'Predict' to get the model's prediction and see a saliency map.
    """)

    uploaded_file = st.file_uploader("Choose a satellite image...", type="tif")

    if uploaded_file is not None:
        # Load and preprocess the image
        image = load_and_prep_image(uploaded_file)

        # Convert tensor back to PIL Image for displaying
        image_np = image.squeeze(0).detach().numpy() # squeeze removes dimensions of size 1 from the tensor
        # Remove any singleton dimensions (if image is grayscale, there is an extra singleton dimension)
        image_np = np.squeeze(image_np)
        # Convert the normalized image back to PIL Image
        display_image = Image.fromarray((image_np/65535 * 255).astype(np.uint8))

        # Display the uploaded image with matplotlib
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(display_image, cmap='gray')
        ax.axis('off') # hide the axes
        st.pyplot(fig)

        # Generate some space before the 'Predict' button
        for _ in range(10):
            st.empty()

        if st.button('Predict'):
            with st.spinner('Generating prediction...'):
                # Generate Grad-CAM
                cam_img = Grad_CAM(image.detach(), model)

                # Create a figure with subplots
                fig, ax = plt.subplots(figsize=(7, 5))  # Change the figure size here
                # Display the heatmap
                cax = ax.imshow(cam_img, cmap='jet')
                ax.axis('off') # hide the axes
                # Add a colorbar
                fig.colorbar(cax)
                st.pyplot(fig)

                # Make a prediction
                output = model(image)
                probs = torch.softmax(output, dim = 1)
                _,pred = torch.max(output, dim = 1)
                
                if  pred == 1:
                    st.success("The model predicts that this image likely contains a methane plume.")
                else:
                    st.info("The model predicts that this image likely does not contain a methane plume.")


                st.write(f"Probability Confidence in Prediction: {probs.max().max() * 100:.2f}%")

    st.header("Disclaimer")
    st.markdown("This tool is intended to assist in methane plume detection. However, it does not guarantee 100% accuracy. Always corroborate with other data sources.")

def about_and_contact():
    st.title("About Us / Contact Us")

    st.markdown("""
    ## About Us
    CleanR is a fast-growing start-up specialized in Methane emissions reporting. We're on a mission to reduce methane emissions by providing a clear method for MRV: monitoring, reporting, and verification. Utilizing the power of satellite imagery and deep learning, we're able to detect potential methane leaks, helping us take a step forward in environmental preservation.

    Our team of developers, data scientists, and environmental experts have worked tirelessly to develop our methane emission detection tool. We're committed to using technology to create a more sustainable future.

    ## Our Team
    Meet our talented team of experts who have contributed to the development of our tool:
    """)

    # List of team member names and photos
    team_members = [
        {'name': 'Annabelle Luo', 'photo': 'photos/Anna_photo.jpg'},
        {'name': 'Yasmina Hobeika', 'photo': 'photos/Yasmina_photo.jpg'},
        {'name': 'Antoine Clouté', 'photo': 'photos/Antoine_photo.png'},
        {'name': 'Amine Zaamoun', 'photo': 'photos/Amine_photo.jpg'},
        {'name': 'Karim El Hage', 'photo': 'photos/Karim_photo.jpg'},
        {'name': 'Leonardo Basili', 'photo': 'photos/Leo_photo.jpg'},
        {'name': 'Ali Najem', 'photo': 'photos/Ali_photo.jpg'},
    ]

    # Define the CSS style for the team member photos
    photo_style = """
        object-fit: cover;
        object-position: center;
        border-radius: 50%;
        width: 150px;
        height: 150px;
        margin: 10px;
    """

    # Display team member photos in a row
    st.markdown("<h2 style='text-align:center;'>Our Team</h2>", unsafe_allow_html=True)

    cols = st.columns(len(team_members)) # creating columns
    for i, member in enumerate(team_members):
        with cols[i]:
            st.image(member['photo'], width=150, caption=member['name'])  # set width of the images

    st.header("Contact Us")
    st.markdown("We'd love to hear from you. Whether you have a question about our tool, need assistance, or just want to provide feedback, feel free to get in touch.")

    # Contact information and feedback form
    st.markdown("""
    - Email: info@cleanr.com
    - Address: 3 Av. Bernard Hirsch, 95000 Cergy
    - Phone: +33 7 00 00 00
    """)

    st.header("Feedback or Questions?")
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Your Message")
    if st.button("Send"):
        st.success("Thank you for your message! We'll get back to you as soon as possible.")

# def main():
#     st.sidebar.title("Navigation")
#     selection = st.sidebar.radio("Go to", list(PAGES.keys()))
#     page = PAGES[selection]
#     page()

# if __name__ == "__main__":
#     main()

st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("Go to", ['Home', 'Detection', 'Impact and Use Cases', 'About Us / Contact Us'])

if page == 'Home':
    home()
elif page == 'Detection':
    detection()
elif page == 'Impact and Use Cases':
    impact_and_use_cases()
elif page == 'About Us / Contact Us':
    about_and_contact()
