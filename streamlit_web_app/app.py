import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

def create_model():
    # Load the pre-trained ResNet101 model
    resnet = models.resnet101(pretrained=False)

    # Modify the first layer to accept single-channel grayscale images
    resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Modify the last fully connected layer for binary classification with softmax activation
    num_classes = 2  # 2 classes: 1 or 0
    resnet.fc = torch.nn.Sequential(
        torch.nn.Linear(resnet.fc.in_features, num_classes)
    )

    return resnet

# Define the model architecture
model = create_model()

# Load the state dictionary
model.load_state_dict(torch.load('models/resnet_model.pth', map_location=torch.device('cpu')))

# Set the model to eval mode
model.eval()

# Hook function containers
activation = []
grad = []

# To get feature maps
def forward_hook(module, input, output):
    activation.append(output)

# To get gradients
def backward_hook(module, grad_in, grad_out):
    grad.append(grad_out[0])

# Registering the hooks on the final convolutional layer of your model
final_layer = model._modules.get('layer4')  # adjust this according to your model architecture
final_layer.register_forward_hook(forward_hook)
final_layer.register_backward_hook(backward_hook)

# Create a function to load and preprocess the image
def load_and_prep_image2(image):
    # Open the image file
    image = Image.open(image)

    # Define the transformation - convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),  # normalize using settings for grayscale images
    ])

    # Apply transformation and add an extra dimension for batch
    image = transform(image).unsqueeze(0)
    
    return image

def load_and_prep_image(image):
    # Open the image file
    image = Image.open(image)
    
    # Convert the image to float
    image = image.convert('F')

    # Define the transformation - convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),  # normalize using settings for grayscale images
    ])

    # Apply transformation and add an extra dimension for batch
    image = transform(image).unsqueeze(0)
    
    return image

# Grad-CAM function
def Grad_CAM(input_img, model):
    activation.clear()  # clear previous activations if any
    grad.clear()  # clear previous gradients if any

    # Forward pass to get activations
    out = model(input_img.view(1, 1, 64, 64))

    # Get the score for the target class (assuming index 1 is the positive class)
    score = out[0, 1]

    # Clear the gradients
    model.zero_grad()

    # Backward pass to get gradients
    score.backward()

    # get the gradients and activations collected in the hook
    grads = grad[0].data.numpy().squeeze()
    fmap = activation[0].data.numpy().squeeze()

    # Calculating cam
    tmp = grads.reshape([grads.shape[0], -1])
    weights = np.mean(tmp, axis = 1)
    
    cam = np.zeros(grads.shape[1:])
    for i, w in enumerate(weights):
        cam += w * fmap[i, :, :]
    
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    
    # Adding heatmap to the original picture
    npic = np.array(input_img).squeeze()
    npic = cv2.cvtColor(npic,cv2.COLOR_GRAY2RGB)  # convert grayscale image to 3-channel grayscale
    cam = cv2.resize(cam, (npic.shape[1], npic.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    cam_img = npic * 1 + heatmap * 0.4  # change the overlay formula to suit grayscale image
    
    return (cam_img)

def detection():
    st.header("Detection")
    st.sidebar.header("Upload Your Image")
    uploaded_file = st.sidebar.file_uploader("Choose a grayscale .tif file", type=['tif'])

    if uploaded_file is not None:
        image = load_and_prep_image(uploaded_file)
        image = image - image.min()
        image /= image.max()

        if st.button('Predict'):
            with st.spinner('Generating prediction...'):
                # Generate Grad-CAM
                cam_img = Grad_CAM(image.detach(), model)
                plt.imshow(cam_img, cmap='hot')
                plt.axis('off')
                st.pyplot(plt)

                # Make a prediction
                output = model(image)

            # Postprocess the prediction
            #prediction = torch.sigmoid(output).item()
            prediction = torch.sigmoid(output)[0, 1].item()

            if prediction > 0.5:
                st.success("The model predicts that this image likely contains a methane plume.")
            else:
                st.info("The model predicts that this image likely does not contain a methane plume.")

            st.write(f"Confidence: {prediction * 100:.2f}%")

        # Convert to numpy and transpose for display
        image = image.squeeze(0).detach().numpy()
        image = np.transpose(image, (1, 2, 0))  # move the channel to the last dimension
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")

    st.sidebar.header("Contact Us")
    st.sidebar.text("For any inquiries or feedback, please contact us at: info@cleanr.com")

    st.sidebar.header("Disclaimer")
    st.sidebar.text("This tool is intended to assist in methane plume detection. However, it does not guarantee 100% accuracy. Always corroborate with other data sources.")

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

    st.image('path_to_some_image', use_column_width=True, caption="Caption for your image")

st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("Go to", ['Home', 'Detection'])

if page == 'Home':
    home()
elif page == 'Detection':
    detection()