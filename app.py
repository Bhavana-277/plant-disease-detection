import streamlit as st
import matlab.engine
import tempfile
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection & Severity Analysis")
st.markdown("Upload a leaf image to detect disease, severity level and treatment recommendation.")

# Start MATLAB Engine only once
@st.cache_resource
def start_engine():
    return matlab.engine.start_matlab()

eng = start_engine()

# Upload image
uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_file.name)

    if st.button("🔍 Predict Disease"):

        with st.spinner("Analyzing leaf image... Please wait..."):

            disease, severity, percent, solution, leaf_mask, disease_mask = eng.predict(
                temp_file.name,
                nargout=6
            )

        st.success("✅ Analysis Completed")

        st.divider()

        # ---------------- Prediction Section ----------------
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**🦠 Disease:** {disease}")

        with col2:
            st.markdown(f"**🌡 Severity Level:** {severity}")

        # Convert percent to float (important)
        percent = float(percent)

        st.write("### Infection Percentage")
        st.progress(min(int(percent), 100))

        if percent < 10:
            st.success(f"Mild Infection ({round(percent,2)}%)")
        elif percent < 30:
            st.warning(f"Moderate Infection ({round(percent,2)}%)")
        else:
            st.error(f"Severe Infection ({round(percent,2)}%)")

        st.divider()

        # ---------------- Treatment Section ----------------
        st.subheader("💊 Treatment Recommendation")
        st.info(solution)

        st.divider()

        # ---------------- Segmentation Section ----------------
        st.subheader("🧠 Segmentation Results")

        # Convert MATLAB arrays to NumPy arrays
        leaf_mask_np = np.array(leaf_mask)
        disease_mask_np = np.array(disease_mask)

        col1, col2 = st.columns(2)

        with col1:
            st.write("🌿 Leaf Mask")
            st.image(leaf_mask_np, use_column_width=True)

        with col2:
            st.write("🦠 Disease Mask")
            st.image(disease_mask_np, use_column_width=True)

        st.divider()

        st.caption("Developed using MATLAB CNN + Streamlit Web Deployment")
