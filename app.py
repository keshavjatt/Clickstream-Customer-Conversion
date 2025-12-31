# app.py (UPDATED FOR NEXT PAGE PREDICTION)
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="E-Shop Conversion Predictor",
    page_icon="🛒",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        saved_data = joblib.load("model/model.pkl")
        return saved_data
    except FileNotFoundError:
        st.error("❌ Model file not found. Please run train_model.py first.")
        st.stop()

saved_data = load_model()
model = saved_data['model']
features = saved_data['features']
label_encoders = saved_data.get('label_encoders', {})

st.title("🛒 E-Shop NEXT PAGE Conversion Predictor")
st.markdown("""
**Predict whether a customer's NEXT click will lead to checkout (page 4 or 5)**  
*Based on current page and session behavior*
""")

# Sidebar with test buttons
with st.sidebar:
    st.header("Quick Test Presets")
    
    if st.button("HIGH Conversion Scenario", use_container_width=True):
        st.session_state.page = 3
        st.session_state.order = 3
        st.session_state.main_category = 4
        st.session_state.country = 29
        st.session_state.price = 45
        st.session_state.colour = 2
        st.session_state.model_photo = 1
        st.session_state.price_2 = 2
        st.session_state.location = 2
        st.session_state.month = 4
        st.session_state.day = 5
        st.rerun()
    
    if st.button("LOW Conversion Scenario", use_container_width=True):
        st.session_state.page = 1
        st.session_state.order = 45
        st.session_state.main_category = 1
        st.session_state.country = 12
        st.session_state.price = 95
        st.session_state.colour = 10
        st.session_state.model_photo = 2
        st.session_state.price_2 = 1
        st.session_state.location = 4
        st.session_state.month = 8
        st.session_state.day = 30
        st.rerun()
    
    st.header("Model Info")
    if 'test_accuracy' in saved_data:
        st.metric("Test Accuracy", f"{saved_data['test_accuracy']*100:.1f}%")
    st.info("Predicts: Will NEXT page be checkout (4 or 5)?")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 2
if 'order' not in st.session_state:
    st.session_state.order = 3
if 'main_category' not in st.session_state:
    st.session_state.main_category = 4
if 'country' not in st.session_state:
    st.session_state.country = 29
if 'price' not in st.session_state:
    st.session_state.price = 45
if 'colour' not in st.session_state:
    st.session_state.colour = 2
if 'model_photo' not in st.session_state:
    st.session_state.model_photo = 1
if 'price_2' not in st.session_state:
    st.session_state.price_2 = 2
if 'location' not in st.session_state:
    st.session_state.location = 2
if 'month' not in st.session_state:
    st.session_state.month = 4
if 'day' not in st.session_state:
    st.session_state.day = 5

# Main interface
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Current Status")
    page = st.slider("Current Page (1-3)", 1, 3, st.session_state.page)
    order = st.slider("Click # in Session", 1, 50, st.session_state.order)
    
    st.subheader("Price")
    price = st.slider("Product Price ($)", 10, 100, st.session_state.price)
    price_2 = st.selectbox(
        "Price Above Average?",
        [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No",
        index=1 if st.session_state.price_2 == 2 else 0
    )

with col2:
    st.subheader("Product")
    main_category = st.selectbox(
        "Main Category",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Trousers",
            2: "Skirts",
            3: "Blouses",
            4: "Sale (BEST)"
        }[x],
        index=st.session_state.main_category - 1
    )
    
    colour = st.selectbox(
        "Color",
        list(range(1, 15)),
        format_func=lambda x: {
            1: "Beige", 2: "Black", 3: "Blue", 4: "Brown",
            5: "Burgundy", 6: "Gray", 7: "Green", 8: "Navy Blue",
            9: "Many Colors", 10: "Olive", 11: "Pink", 12: "Red",
            13: "Violet", 14: "White"
        }[x],
        index=st.session_state.colour - 1
    )
    
    location = st.selectbox(
        "Photo Location",
        list(range(1, 7)),
        format_func=lambda x: {
            1: "Top Left",
            2: "Top Middle",
            3: "Top Right",
            4: "Bottom Left",
            5: "Bottom Middle",
            6: "Bottom Right"
        }[x],
        index=st.session_state.location - 1
    )

with col3:
    st.subheader("Customer & Time")
    country = st.selectbox(
        "Customer Country",
        options=list(range(1, 48)),
        format_func=lambda x: {
            1: "Australia", 2: "Austria", 3: "Belgium", 
            29: "Poland", 42: "USA", 12: "Unidentified",
            16: "Germany", 15: "France"
        }.get(x, f"Country {x}"),
        index=28 if st.session_state.country == 29 else (st.session_state.country - 1)
    )
    
    model_photo = st.selectbox(
        "Model Pose",
        [1, 2],
        format_func=lambda x: "En Face" if x == 1 else "Profile",
        index=0 if st.session_state.model_photo == 1 else 1
    )
    
    month = st.selectbox(
        "Month",
        [4, 5, 6, 7, 8],
        format_func=lambda x: {
            4: "April", 5: "May", 6: "June",
            7: "July", 8: "August"
        }[x],
        index=st.session_state.month - 4
    )
    
    day = st.slider("Day of Month", 1, 31, st.session_state.day)

# Prepare input data
input_data = {
    'page': page,
    'order': order,
    'price': price,
    'price 2': price_2,
    'month': month,
    'day': day,
    'country': country,
    'page 1 (main category)': main_category,
    'colour': colour,
    'location': location,
    'model photography': model_photo
}

# Create dataframe
input_df = pd.DataFrame([input_data])

# Apply preprocessing
for col, le in label_encoders.items():
    if col in input_df.columns:
        val = str(input_df[col].iloc[0])
        if val in le.classes_:
            input_df.loc[:, col] = le.transform([val])
        else:
            input_df.loc[:, col] = le.transform([le.classes_[0]])

# Ensure correct feature order
input_df = input_df[features]

# Always show prediction
st.markdown("---")
st.subheader("Live Prediction")

try:
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    # Display results
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        st.write("### Prediction Result")
        if prediction == 1:
            st.success(f"## NEXT PAGE will be CHECKOUT")
        else:
            st.error(f"## NEXT PAGE will NOT be checkout")
    
    with col_result2:
        st.write("### Probability")
        conv_prob = probability[1] * 100
        
        if prediction == 1:
            st.metric(
                "Chance next page is 4/5", 
                f"{conv_prob:.1f}%",
                delta=f"+{conv_prob-50:.1f}%"
            )
        else:
            st.metric(
                "Chance next page is 4/5", 
                f"{conv_prob:.1f}%",
                delta=f"{conv_prob-50:.1f}%"
            )
        
        # Progress bar
        st.progress(conv_prob/100, 
                   text=f"Conversion Probability: {conv_prob:.1f}%")
        
        # Interpretation
        if conv_prob > 70:
            st.info("**HIGH** chance of reaching checkout next")
        elif conv_prob > 40:
            st.warning("**MODERATE** chance of reaching checkout next")
        else:
            st.info("**LOW** chance of reaching checkout next")
    
    # Feature importance for this prediction
    st.markdown("---")
    st.subheader("Key Factors Influencing This Prediction")
    
    if 'feature_importance' in saved_data:
        feature_importance = saved_data['feature_importance'].head(5)
        
        for idx, row in feature_importance.iterrows():
            feat_name = row['feature']
            importance = row['importance'] * 100
            value = input_data.get(feat_name, "")
            
            # Format value display
            if feat_name == 'page':
                value_disp = f"Page {value}"
            elif feat_name == 'page 1 (main category)':
                value_disp = {1: "Trousers", 2: "Skirts", 3: "Blouses", 4: "Sale"}.get(value, value)
            elif feat_name == 'colour':
                value_disp = {
                    1: "Beige", 2: "Black", 3: "Blue", 4: "Brown",
                    5: "Burgundy", 6: "Gray", 7: "Green", 8: "Navy Blue",
                    9: "Many Colors", 10: "Olive", 11: "Pink", 12: "Red",
                    13: "Violet", 14: "White"
                }.get(value, value)
            elif feat_name == 'location':
                value_disp = {
                    1: "Top Left", 2: "Top Middle", 3: "Top Right",
                    4: "Bottom Left", 5: "Bottom Middle", 6: "Bottom Right"
                }.get(value, value)
            elif feat_name == 'model photography':
                value_disp = "En Face" if value == 1 else "Profile"
            elif feat_name == 'price 2':
                value_disp = "Yes" if value == 1 else "No"
            else:
                value_disp = str(value)
            
            st.write(f"**{feat_name}**: {value_disp} (Impact: {importance:.1f}%)")
            
except Exception as e:
    st.error(f"❌ Prediction error: {str(e)}")