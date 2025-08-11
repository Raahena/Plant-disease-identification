import streamlit as st
import tensorflow as tf
import numpy as np
import json
import time
import uuid
from pathlib import Path
import os

# Setup shared paths
SHARED_DIR = Path("shared")
QUEUE_FILE = SHARED_DIR / "disease_queue.json"
RESULTS_FILE = SHARED_DIR / "recommendations.json"

# Make sure shared directory exists
SHARED_DIR.mkdir(exist_ok=True)
if not QUEUE_FILE.exists():
    with open(QUEUE_FILE, "w") as f:
        json.dump({"requests": [], "processed": []}, f)
if not RESULTS_FILE.exists():
    with open(RESULTS_FILE, "w") as f:
        json.dump({}, f)

# Function to load model and predict
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Function to request recommendation from Gemini service
def request_recommendation(disease_name):
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    # Create request
    request = {
        "id": request_id,
        "disease": disease_name,
        "timestamp": time.time()
    }
    
    # Load current queue
    if QUEUE_FILE.exists():
        with open(QUEUE_FILE, "r") as f:
            queue_data = json.load(f)
    else:
        queue_data = {"requests": [], "processed": []}
    
    # Add request to queue
    queue_data["requests"].append(request)
    
    # Save updated queue
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue_data, f)
    
    return request_id

# Function to check if recommendation is ready
def get_recommendation(request_id):
    if not RESULTS_FILE.exists():
        return None
    
    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)
    
    return results.get(request_id)

# CSS for styling
def load_css():
    st.markdown("""
    <style>
    .image-guidelines {
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 10px 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    .verification-box {
        background-color: #e7f3fe;
        border-left: 4px solid #2196F3;
        padding: 15px;
        margin: 20px 0;
        border-radius: 5px;
    }
    .redirect-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        border: none;
    }
    .about-section {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .shop-button {
        background-color: #ff9800;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition by Image", "Disease Recognition by Symptoms", "Chatbot", "Contact Expert", "Shop"])

# Load CSS
load_css()

if app_mode == 'Home':
    st.header("PLANT CARE 🌱")
    image_path = "home.jpg"
    st.image(image_path, width=200)
    st.markdown("""
    Welcome to the Plant Care 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Need any help? Ask our chatbot. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition by Image** page and upload an image of a plant with suspected diseases.
    2. **Describe Symptoms:** Alternatively, use our **Disease Recognition by Symptoms** tool to identify diseases based on visual symptoms.
    3. **Analysis:** Our system will process your information using advanced algorithms to identify potential diseases.
    4. **Results:** View the results and recommendations for treatment and supplements.

    ### Why Choose Us?
    - **Multiple Detection Methods:** Use image recognition or symptom description for accurate diagnosis.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on one of our disease recognition tools in the sidebar to start diagnosing your plants!

    ### Available Plants for Disease Detection
    Here is a list of plants that our system can recognize and diagnose:
    """)

    # Display the plant list
    st.write("""
    Here is a list of plants that our system can recognize and diagnose:

    - Apple 🍎  
    - Blueberry 🫐  
    - Cherry 🍒  
    - Corn 🌽  
    - Grape 🍇  
    - Orange 🍊  
    - Peach 🍑  
    - Pepper 🌶️  
    - Potato 🥔  
    - Raspberry 🍓  
    - Soybean 🌱  
    - Squash 🥒  
    - Strawberry 🍓  
    - Tomato 🍅  
    """)

    # About section (moved from About page)
    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    st.header("About Dataset")
    st.markdown("""
        This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
        This dataset consists of about 87K RGB images of healthy and diseased crop leaves which is categorized into 38 different classes.
        The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
        A new directory containing 33 test images is created later for prediction purposes.

        #### Content
        1. train (70,295 images)
        2. test (33 images)
        3. validation (17,572 images)
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
elif app_mode == "Disease Recognition by Image":
    st.title("🌿 Plant Disease Recognition by Image")
    
    # Image guidelines
    st.markdown('<div class="image-guidelines">', unsafe_allow_html=True)
    st.subheader("📸 Image Guidelines")
    st.markdown("""
    For best results, please follow these guidelines when taking photos:
    - Use a **clear, well-lit photo** with good focus
    - Include a **single leaf** in the frame when possible
    - Use a **plain background** (white or light-colored is best)
    - Capture the **entire leaf** including any discoloration, spots, or unusual patterns
    - Avoid shadows or reflections that might obscure symptoms
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Upload Image
    st.subheader("📤 Upload a Plant Image")
    test_image = st.file_uploader("Upload an image of a plant leaf for disease recognition", type=["jpg", "jpeg", "png"])

    if 'current_disease' not in st.session_state:
        st.session_state['current_disease'] = None

    if test_image:
        with st.expander("📷 Preview Uploaded Image", expanded=True):
            st.image(test_image, use_column_width=True)

        # Prediction & Recommendation tabs
        tab1, tab2, tab3 = st.tabs(["🔎 Prediction", "💊 Recommendations", "✅ Verification"])

        with tab1:
            if st.button("Predict Disease"):
                with st.spinner("Analyzing the image..."):
                    st.snow()
                    result_index = model_prediction(test_image)
                    class_name = [
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy'
                    ]
                    disease_name = class_name[result_index]
                    st.session_state['disease'] = disease_name

                    if st.session_state['current_disease'] != disease_name:
                        st.session_state['current_disease'] = disease_name
                        st.session_state.pop('recommendation_id', None)

                    st.success(f"✅ Prediction: **{disease_name}**")
                    
                    # Add verification prompt
                    st.info("Please verify this prediction using the 'Verification' tab or our Symptom-based recognition tool.")

        with tab2:
            if 'disease' in st.session_state:
                disease_name = st.session_state['disease']

                if 'recommendation_id' not in st.session_state:
                    st.session_state['recommendation_id'] = request_recommendation(disease_name)
                    st.session_state['recommendation_request_time'] = time.time()

                recommendation_section = st.empty()
                max_retries = 10
                retry_count = 0
                recommendation = None

                while retry_count < max_retries and not recommendation:
                    recommendation = get_recommendation(st.session_state['recommendation_id'])
                    if recommendation:
                        break
                    retry_count += 1
                    time.sleep(1)

                    with recommendation_section.container():
                        st.info(f"Generating treatment recommendations for **{disease_name}**...")
                        st.progress(retry_count / max_retries)
                        if retry_count >= max_retries // 2:
                            st.caption("⏳ This may take a few more seconds...")

                with recommendation_section.container():
                    if recommendation:
                        st.success("✅ Recommendations Ready!")
                        st.subheader("💡 Treatment Recommendations")
                        st.markdown(recommendation["recommendation"])
                    else:
                        st.warning("Still working on recommendations...")
                        if st.button("🔁 Retry"):
                            st.experimental_rerun()
                            
        with tab3:
            st.subheader("Verify Your Results")
            st.markdown("""
            It's important to verify the AI prediction with symptoms you observe:
            
            1. **Check if the visual symptoms match** the predicted disease
            2. **Confirm with symptom-based diagnosis** for more accuracy
            
            If the image-based prediction and symptom-based diagnosis don't match, we recommend:
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Use Symptom-Based Recognition"):
                    st.markdown(
                        f'<a href="https://bloom-wise-garden-46.lovable.app/" target="_blank" class="redirect-button">Go to Symptom-Based Tool</a>',
                        unsafe_allow_html=True
                    )
                    
            with col2:
                if st.button("Contact Expert"):
                    st.markdown(
                        f'<a href="https://bloom-wise-garden-46.lovable.app/experts" target="_blank" class="redirect-button">Contact Expert</a>',
                        unsafe_allow_html=True
                    )
                    
            if 'disease' in st.session_state:
                st.markdown('<div class="verification-box">', unsafe_allow_html=True)
                st.subheader("Do the results match what you observe?")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, it matches"):
                        st.success("Great! You can proceed with the recommendations.")
                with col2:
                    if st.button("No, it doesn't match"):
                        st.warning("Please try our symptom-based tool or contact an expert.")
                        
                st.markdown('</div>', unsafe_allow_html=True)

elif app_mode == "Disease Recognition by Symptoms":
    st.title("🌿 Plant Disease Recognition by Symptoms")
    
    st.markdown("""
    Use our symptom-based diagnosis tool to identify plant diseases by describing what you observe.
    """)
    
    st.info("Click the button below to access our symptom-based diagnosis tool")
    
    if st.button("🔍 Start Symptom-Based Diagnosis", use_container_width=True):
        st.markdown(
            f'<a href="https://bloom-wise-garden-46.lovable.app/checker" target="_blank" class="redirect-button">Open Symptom-Based Tool</a>',
            unsafe_allow_html=True
        )
        
    st.markdown("""
    ### Why use symptom-based diagnosis?
    
    - Provides a complementary approach to image recognition
    - Helps verify results from image analysis
    - Guides you through specific symptoms to look for
    - Works even when you don't have a clear photo
    
    After completing the symptom-based diagnosis, return here to compare results with image recognition or proceed to treatment recommendations.
    """)
    
    # Option to return to image-based diagnosis
    if st.button("Return to Image-Based Diagnosis"):
        st.session_state['app_mode'] = "Disease Recognition by Image"
        st.experimental_rerun()

elif app_mode == "Chatbot":
    st.title("🤖 AI Plant Care Assistant")
    
    st.markdown("""
    Chat with our AI assistant about plant diseases, care tips, or any gardening questions you have.
    """)
    
    # Create a button to redirect to external chat
    if st.button("Start Chatting", use_container_width=True):
        st.markdown(
            f'<a href="https://bloom-wise-garden-46.lovable.app/chat" target="_blank" class="redirect-button">Open Chat Assistant</a>',
            unsafe_allow_html=True
        )
    
    st.markdown("""
    ### Why use our chat assistant?
    
    - Get immediate answers to your plant care questions
    - Learn more about specific plant diseases and treatments
    - Receive personalized gardening advice
    - Available 24/7 for all your plant-related queries
    
    If you need more specialized help, consider contacting one of our experts.
    """)
    
    # Show context from disease detection if available
    if 'disease' in st.session_state:
        st.info(f"You recently identified a plant with **{st.session_state['disease']}**. Our chat assistant can provide more information about this condition.")
    
    # Option to contact expert
    if st.button("Need Expert Help?"):
        st.markdown(
            f'<a href="https://bloom-wise-garden-46.lovable.app/experts" target="_blank" class="redirect-button">Contact Expert</a>',
            unsafe_allow_html=True
        )

elif app_mode == "Contact Expert":
    st.title("📞 Contact an Expert")
    st.markdown("""
    Need expert advice? Connect with our agricultural specialists for personalized guidance and support.
    """)
    
    # Button to redirect to expert page
    if st.button("Connect with Experts Now", use_container_width=True):
        st.markdown(
            f'<a href="https://bloom-wise-garden-46.lovable.app/experts" target="_blank" class="redirect-button">Contact Our Experts</a>',
            unsafe_allow_html=True
        )
    
    st.subheader("When to Contact an Expert")
    st.markdown("""
    We recommend contacting our experts when:
    
    - Image recognition and symptom-based diagnosis give different results
    - Your plant shows unusual symptoms not covered in our database
    - You need personalized treatment recommendations
    - The condition is severe or spreading rapidly
    - You need on-site inspection or laboratory testing
    """)
    
    # Display expert profiles
    st.subheader("Our Expert Team")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Dr. Sarah Johnson**  
        *Plant Pathologist*  
        Specializes in fungal diseases and organic treatment methods
        """)
        
    with col2:
        st.markdown("""
        **Prof. Michael Chen**  
        *Agricultural Entomologist*  
        Expert in pest management and integrated pest control strategies
        """)
        
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        **Dr. Elena Rodriguez**  
        *Horticultural Scientist*  
        Focuses on sustainable farming and plant nutrition
        """)
        
    with col4:
        st.markdown("""
        **James Wilson**  
        *Master Gardener*  
        Specializes in home gardening and ornamental plants
        """)
        
    st.markdown("""
    You can also ask initial questions using our **Chatbot** for instant guidance.
    """)
    
    if st.button("Try Our Chatbot"):
        st.markdown(
            f'<a href="https://bloom-wise-garden-46.lovable.app/chat" target="_blank" class="redirect-button">Open Chat Assistant</a>',
            unsafe_allow_html=True
        )

elif app_mode == "Shop":
    st.title("🛒 Plant Care Shop")
    
    st.markdown("""
    Welcome to our Plant Care Shop! Find everything you need for healthy plants:
    
    - **Organic treatments** for plant diseases
    - **Prevention products** to keep your plants healthy
    - **Gardening tools** and equipment
    - **Seeds and plants** for your garden
    - **Educational resources** about plant care
    """)
    
    # Featured products section
    st.subheader("Featured Products")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Organic Fungicide**  
        Natural solution for fungal diseases
        """)
    
    with col2:
        st.markdown("""
        **Professional Pruning Shears**  
        Essential for healthy plant maintenance
        """) 
    
    with col3:
        st.markdown("""
        **Complete Soil Test Kit**  
        Check soil health and nutrient levels
        """)
    
    # Shop button
    st.markdown("<div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
    st.markdown(
        f'<a href="https://bloom-wise-garden-46.lovable.app/shop" target="_blank" class="shop-button">Visit Our Shop</a>',
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Disease-specific products
    if 'disease' in st.session_state:
        st.subheader("Recommended Products")
        st.info(f"Based on your plant's condition (**{st.session_state['disease']}**), we recommend these products:")
        
        st.markdown("""
        - **Specialized Treatment Kit**
        - **Prevention Spray**
        - **Plant Nutrition Supplement**
        """)
        
        st.markdown(
            f'<a href="https://bloom-wise-garden-46.lovable.app/shop" target="_blank" class="redirect-button">View Recommended Products</a>',
            unsafe_allow_html=True
        )