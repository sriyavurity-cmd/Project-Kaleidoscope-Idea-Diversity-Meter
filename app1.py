import streamlit as st
import pandas as pd
import numpy as np
import base64
import json
import time
import asyncio 
import requests # <-- ADDED: for standard API calls
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.express as px

# --- API KEY & URL (Leave apiKey as "" for Canvas to inject the key) ---
# NOTE: For local testing, you will need to replace "" with your actual Gemini API key, 
# or set it as an environment variable (e.g., GEMINI_API_KEY) and access it via os.environ.
# For now, we will leave it blank for Canvas compatibility, and use standard requests.
API_KEY = "AIzaSyBdAcJ5aJ0Ck2ep-Gpd5F0b_0sn-ehZPfM" 
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

# --- 0. STREAMLIT CONFIGURATION (Vibrant Dark Mode Theme) ---
st.set_page_config(
    page_title="Project Kaleidoscope: Diversity Meter",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for a vibrant look
st.markdown("""
<style>
    /* Main Background and Text Color */
    .stApp {
        background-color: #0d1117; 
        color: #c9d1d9;
    }
    /* Title Styling */
    h1 {
        color: #ff4b4b; /* Streamlit Red */
        text-align: center;
        border-bottom: 2px solid #30363d;
        padding-bottom: 10px;
    }
    /* Subtitle Styling */
    h3 {
        color: #58a6ff; /* Blue for subheaders */
    }
    /* Button Styling */
    div.stButton > button {
        background-color: #58a6ff;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.2s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div.stButton > button:hover {
        background-color: #4a8ee1;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# --- 1. CONFIGURATION AND MODEL LOADING ---
@st.cache_resource
def load_model():
    """Load the sentence embedding model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- 2. CORE FUNCTIONS ---

def calculate_metrics(embeddings):
    """Calculates Diversity Score and Conceptual Radius."""
    n_ideas = len(embeddings)
    
    # 1. Diversity Score (Average Distance)
    if n_ideas < 2:
        return 0.0, 0.0, 0.0
        
    cosine_sim = cosine_similarity(embeddings)
    upper_triangle_indices = np.triu_indices(n_ideas, k=1)
    mean_similarity = cosine_sim[upper_triangle_indices].mean()
    diversity_score = 1.0 - mean_similarity

    # 2. Conceptual Radius (Measure of explored space)
    centroid = np.mean(embeddings, axis=0)
    distances_from_centroid = [1.0 - cosine_similarity([idea], [centroid])[0][0] for idea in embeddings]
    conceptual_radius = np.mean(distances_from_centroid)

    # 3. Freshness Score (Last idea vs. Centroid of all previous ideas)
    if n_ideas > 1:
        previous_embeddings = embeddings[:-1]
        last_embedding = embeddings[-1]
        previous_centroid = np.mean(previous_embeddings, axis=0)
        freshness = 1.0 - cosine_similarity([last_embedding], [previous_centroid])[0][0]
    else:
        freshness = 0.0

    return diversity_score, conceptual_radius, freshness


def run_analysis():
    """Generates embeddings, calculates metrics, and updates the visualization."""
    if not st.session_state.all_ideas:
        st.error("Please submit at least one idea.")
        return

    # Create DataFrame from stored ideas
    ideas_df = pd.DataFrame(st.session_state.all_ideas, columns=['Idea'])
    
    # Generate Embeddings
    with st.spinner("🧠 Analyzing and generating conceptual vectors..."):
        embeddings = model.encode(ideas_df['Idea'].tolist())

    # Calculate all metrics
    diversity_score, conceptual_radius, freshness_score = calculate_metrics(embeddings)
    
    # Store metrics for display
    st.session_state.diversity_score = diversity_score
    st.session_state.conceptual_radius = conceptual_radius
    st.session_state.freshness_score = freshness_score
    
    # Run dimensionality reduction (PCA for stability)
    with st.spinner("🗺️ Creating the Idea Map visualization..."):
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
        ideas_df['X'] = coords[:, 0]
        ideas_df['Y'] = coords[:, 1]
        
        # Plotly chart generation
        fig = px.scatter(ideas_df, x='X', y='Y', 
                         hover_data={'Idea': True, 'X': False, 'Y': False}, 
                         title='Conceptual Idea Map (PCA)',
                         labels={'X': 'PCA Component 1', 'Y': 'PCA Component 2'},
                         color_discrete_sequence=['#ff4b4b'], 
                         template="plotly_dark")

        # Make map interactive and aesthetically pleasing
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color='#58a6ff')))
        fig.update_layout(height=450, margin=dict(t=50, b=50, l=50, r=50))
        
    st.session_state.idea_map_fig = fig
    st.session_state.analysis_run = True


# Function to call the Gemini API for image analysis
# Changed from async to standard sync function using requests
def analyze_image_with_gemini(uploaded_file):
    """Sends an image to the Gemini API to extract text and ideas using standard requests."""
    
    # Check if API_KEY is set for local execution
    if not API_KEY:
        st.error("API Key is missing! For local testing, you must set the 'API_KEY' variable in the code.")
        return []

    # Convert image to base64
    bytes_data = uploaded_file.getvalue()
    base64_encoded_image = base64.b64encode(bytes_data).decode('utf-8')

    # Construct the user prompt and system instruction
    prompt = "Transcribe all distinct ideas, text, and concepts visible in this image (e.g., from a whiteboard or sticky notes). List each idea as a separate item. If you cannot extract clear ideas, describe the image content briefly."
    
    system_instruction = "You are an idea extraction expert. Your output MUST be a JSON array of strings, where each string is a single, concise idea or transcribed text fragment. Do not include any explanation or markdown outside the JSON array."
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": uploaded_file.type,
                            "data": base64_encoded_image
                        }
                    }
                ]
            }
        ],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            }
        }
    }

    # Exponential backoff for API call
    for i in range(3):
        try:
            # Use standard requests.post instead of st.experimental_rerun_with_fetch
            response = requests.post(
                API_URL, 
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            # Process the response
            result = response.json()
            if result and 'candidates' in result and result['candidates']:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content'] and candidate['content']['parts']:
                    json_text = candidate['content']['parts'][0].get('text')
                    
                    # Parse the JSON array of ideas
                    try:
                        extracted_ideas = json.loads(json_text)
                        # Ensure it's a list/array
                        if isinstance(extracted_ideas, list):
                            return extracted_ideas
                        else:
                             st.error("Gemini returned text that wasn't a valid JSON list. Check API output.")
                             return []
                    except json.JSONDecodeError:
                        st.error(f"Failed to parse JSON from Gemini response: {json_text}")
                        return []
            
            # If no candidates or invalid structure
            st.error("Gemini API call returned an empty or invalid response.")
            return []

        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error during API call: {e}")
            return []
        except Exception as e:
            if i < 2:
                # Wait longer on each retry
                wait_time = 2 ** i
                st.warning(f"API call failed, retrying in {wait_time}s... Error: {e}")
                time.sleep(wait_time) 
                continue
            st.error(f"Error during image analysis: {e}")
            return []
    return []

# --- Function to handle the sync call from the main execution thread ---
def handle_image_analysis(uploaded_file):
    """Runs the image analysis function and processes results."""
    
    # Check cache first
    # NOTE: When running locally, uploaded_file.file_id might not be reliable, but we'll keep it for the concept.
    # For robust local caching, you might hash the file contents.
    file_id = uploaded_file.file_id if hasattr(uploaded_file, 'file_id') else uploaded_file.name 
    
    if file_id in st.session_state.extracted_ideas_cache:
        extracted_list = st.session_state.extracted_ideas_cache[file_id]
        
    else:
        # Start the analysis
        with st.spinner("🚀 Analyzing image with Gemini API to extract ideas..."):
            # Now calling the standard sync function
            extracted_list = analyze_image_with_gemini(uploaded_file)
        
        # Cache the result
        if extracted_list:
             st.session_state.extracted_ideas_cache[file_id] = extracted_list
        

    if extracted_list:
        # Filter and add unique ideas
        current_ideas = set(st.session_state.all_ideas)
        new_unique_ideas = [
            idea for idea in extracted_list 
            if idea not in current_ideas
        ]
        
        if new_unique_ideas:
            st.session_state.all_ideas.extend(new_unique_ideas)
            run_analysis()
            st.toast(f"Added {len(new_unique_ideas)} ideas extracted from image!")
        else:
            st.warning("All ideas extracted from the image were already in the team bank.")
    else:
        st.error("Could not extract clear ideas from the image. Please try a clearer image.")


# --- 3. STREAMLIT FRONTEND LAYOUT ---

st.title("💡 Project Kaleidoscope: Idea Diversity Meter")
st.markdown("### Gamified Innovation: Challenge Repetitive Thinking")

# Initialize state variables
if 'all_ideas' not in st.session_state:
    st.session_state.all_ideas = []
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'diversity_score' not in st.session_state:
    st.session_state.diversity_score = 0.0
if 'conceptual_radius' not in st.session_state:
    st.session_state.conceptual_radius = 0.0
if 'freshness_score' not in st.session_state:
    st.session_state.freshness_score = 0.0
if 'idea_map_fig' not in st.session_state:
    st.session_state.idea_map_fig = None
if 'extracted_ideas_cache' not in st.session_state:
    st.session_state.extracted_ideas_cache = {}


# Idea Submission Input
with st.container(border=True):
    st.markdown("### Submission Dock")
    
    col_text, col_upload = st.columns([1, 1])

    with col_text:
        st.markdown("#### 1. Text Ideas")
        
        # Use a new key 'text_input_area' for the text area itself
        # The value will be stored in st.session_state.text_input_area
        new_idea_text = st.text_area(
            "Enter multiple ideas, one per line:", 
            height=120, 
            key="text_input_area", # NEW KEY for the widget
            placeholder="E.g., \nDesign a sustainable solar panel.\nLaunch an interactive learning app.",
            label_visibility="collapsed"
        )
        
        # Button to process the ideas
        if st.button("Submit Text & Analyze", use_container_width=True, key="submit_btn"):
            submitted_ideas = [i.strip() for i in new_idea_text.split('\n') if i.strip()]

            if submitted_ideas:
                new_unique_ideas = [
                    idea for idea in submitted_ideas 
                    if idea not in st.session_state.all_ideas
                ]

                if new_unique_ideas:
                    st.session_state.all_ideas.extend(new_unique_ideas)
                    run_analysis()
                    st.toast(f"Added {len(new_unique_ideas)} new text idea(s) for analysis!")
                    
                    # Clear the text input after submission using its key
                    # This must be done AFTER run_analysis() to prevent the RERUN from clearing the ideas prematurely.
                    st.session_state.text_input_area = "" 
                else:
                    st.warning("All submitted text ideas were already in the team bank.")
            else:
                st.error("Please enter at least one idea.")
    
    with col_upload:
        st.markdown("#### 2. Image Ideas (Whiteboard/Notes)")
        uploaded_file = st.file_uploader(
            "Upload image to extract ideas:", 
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="image_uploader" # Added key for better state management
        )
        
        # The button now checks if a file is present and calls the handler function
        if st.button("Analyze Image Ideas", use_container_width=True, key="analyze_image_btn"):
            if uploaded_file is not None:
                # Call the synchronous wrapper which handles async execution and caching
                handle_image_analysis(uploaded_file)
            else:
                st.error("Please upload an image file first.")


# Analysis and Visualization Section
if st.session_state.analysis_run:
    st.markdown("---")
    st.markdown("## 📊 Innovation Dashboard")
    
    # METRICS ROW
    col_score, col_radius, col_freshness = st.columns(3)

    # 1. Diversity Score
    with col_score:
        st.subheader("Team Diversity Score")
        score = st.session_state.diversity_score
        st.metric(
            label="Conceptual Spread (0.0 to 1.0)",
            value=f"{score:.3f}",
            delta_color="off"
        )
        if score < 0.2:
            st.error("🚨 STAGNATION ALERT! Ideas are too similar. Try a new persona!")
        elif score < 0.4:
            st.warning("⚠️ Low Diversity. Push boundaries!")
        else:
            st.success("✅ Good conceptual diversity!")

    # 2. Conceptual Radius
    with col_radius:
        st.subheader("Conceptual Radius")
        radius = st.session_state.conceptual_radius
        st.metric(
            label="Explored Idea Space",
            value=f"{radius:.3f}",
            delta_color="off"
        )
        st.caption("Average distance of all ideas from the team's conceptual center.")

    # 3. Freshness Score (Gamified Feature)
    with col_freshness:
        st.subheader("Freshness Challenge")
        freshness = st.session_state.freshness_score
        # Compare against a decent baseline of 0.2
        delta_val = freshness - 0.2 
        
        st.metric(
            label="New Idea vs. Old Average",
            value=f"{freshness:.3f}",
            delta=f"{'+' if delta_val > 0 else ''}{delta_val:.3f}",
            delta_color="normal" if delta_val > 0 else "inverse"
        )
        if freshness > 0.4:
            st.balloons()
            st.success("✨ BREAKTHROUGH! You successfully challenged repetitive thinking.")
        elif freshness < 0.1:
            st.error("🎯 REPETITION! Your new idea is very close to the team's average.")
        elif freshness < 0.2:
            st.warning("⬆️ Routine Thinking. Try a concept from a different domain.")

    # MAP AND TEAM IDEAS
    col_map, col_ideas = st.columns([7, 3])

    with col_map:
        st.markdown("### 🗺️ Conceptual Idea Map (PCA)")
        st.plotly_chart(st.session_state.idea_map_fig, use_container_width=True)
        st.caption("Each point is an idea. Closeness = conceptual similarity.")

    with col_ideas:
        st.markdown("### 👥 Team Idea Bank")
        # Display the ideas as a numbered list
        idea_list_markdown = ""
        for i, idea in enumerate(st.session_state.all_ideas):
            # Highlight the newest idea 
            if i == len(st.session_state.all_ideas) - 1 and len(st.session_state.all_ideas) > 1:
                idea_list_markdown += f"**{i+1}.** *{idea}* (NEW)\n"
            else:
                idea_list_markdown += f"{i+1}. {idea}\n"
        
        st.markdown(idea_list_markdown)

    # Reset Button for starting a new session
    if st.button("Start New Brainstorm Session"):
        st.session_state.all_ideas = []
        st.session_state.analysis_run = False
        st.session_state.diversity_score = 0.0
        st.session_state.conceptual_radius = 0.0
        st.session_state.freshness_score = 0.0
        if 'extracted_ideas_cache' in st.session_state:
            st.session_state.extracted_ideas_cache = {}
        st.experimental_rerun()
