import streamlit as st
import sys
import os
import base64
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Backend connection
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Page configuration
st.set_page_config(
    page_title="AeroPulse AI", 
    page_icon="aircraft.jpg", 
    layout="wide"
)

# ===== API KEY MANAGEMENT =====
env_key = os.getenv("GROQ_API_KEY")
api_key = env_key if env_key else None

# ===== AGENT INITIALIZATION WITH CACHING =====
@st.cache_resource
def get_agent(key):
    """Cached agent initialization"""
    try:
        from agents import AdvisorAgent
        return AdvisorAgent(key)
    except ImportError as e:
        st.error(f"Import Error: {e}")
        return None
    except Exception as e:
        st.error(f"Agent Initialization Error: {e}")
        return None

if not api_key:
    st.warning("No GROQ_API_KEY found. Add it to .env file for full functionality.")
    agent = None
else:
    agent = get_agent(api_key)
    if agent is None:
        st.error("Failed to initialize agent. Check your setup.")
        st.stop()

# ===== HELPER FUNCTIONS =====
def get_img_as_base64(file_path):
    """Encode image to base64"""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

logo_base64 = get_img_as_base64("aircraft.jpg")
logo_src = f"data:image/jpeg;base64,{logo_base64}" if logo_base64 else "https://cdn-icons-png.flaticon.com/512/2200/2200342.png"
hero_base64 = get_img_as_base64("image.jpg")
hero_src = f"data:image/jpeg;base64,{hero_base64}" if hero_base64 else "https://images.unsplash.com/photo-1556388169-db19adc9608f?q=80&w=2500&auto=format&fit=crop"

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    /* Main Background */
    .stApp { 
        background-color: oklch(0.979 0.021 166.113);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .block-container {
    padding-top: 3rem;
    padding-bottom: 5rem;
    }

    /* Header */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 30px;
        background: oklch(0.95 0.052 163.051);
        border-bottom: 1px solid #e2e8f0;
        margin: 30px 0;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
    
    .header-logo { 
        font-size: 1.6rem;
        font-weight: 800;
        color: oklch(0.262 0.051 172.552);
        display: flex; 
        align-items: center;
        gap: 15px;
    }
            
    .header-logo img {
        height: 50px;
        border-radius: 8px;
    }
    
    /* Hero Section */
    .hero-title { 
        font-size: 3.8rem;
        font-weight: 900;
        color: oklch(0.262 0.051 172.552);
        line-height: 1.1;
        margin-bottom: 20px;
    }
    
    .hero-subtitle { 
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 30px;
        line-height: 1.6;
    }
            
    /* SECTION LABEL */
    .section-div {
        text-align: center;
        margin: 50px 0 20px;
    }
    
    .section-label {
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: oklch(0.596 0.145 163.225);
        background-color: oklch(0.95 0.052 163.051);
        padding: 8px 16px;
        border-radius: 20px;
    }
    
    /* Input Form*/
    [data-testid="stForm"] {
        background-color: oklch(0.905 0.093 164.15);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.06);
    }
            
    .stTextInput input,
    .stDateInput input,
    .stTimeInput input {
        background-color: oklch(0.979 0.021 166.113);
        color: oklch(0.596 0.145 163.225);
    }           
    
    div[data-baseweb="select"] > div {
        background-color: oklch(0.979 0.021 166.113);
    }

    div[data-baseweb="select"] span {
        color: oklch(0.596 0.145 163.225);
    }

    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"]:focus-within {
        border-color: #1565c0 !important;
    }

    /* Buttons */
    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: oklch(0.596 0.145 163.225);
        color: white !important;
        border: none !important;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%; 
    }
            
    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: oklch(0.508 0.118 165.612);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        transform: translateY(-1px);
    }
    
    /* Result Card */
    .result-card { 
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin: 30px 0px 30px;
        animation: fadeIn 0.6s ease-out;       
    }

    /* RAG Analysis */
    .rag-analysis {
        background: oklch(0.905 0.093 164.15);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 4px solid oklch(0.262 0.051 172.552);
    }
    
    /* FACTORS TITLE */
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: oklch(0.262 0.051 172.552);
        margin: 25px 0 12px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* KEY FACTORS */
    .factor-badge {
        display: inline-block;
        background: #ffffff;
        color: oklch(0.262 0.051 172.552);
        padding: 6px 12px;
        border-radius: 8px;
        margin: 5px 5px 5px 0;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 10px 40px rgba(0,0,0,0.06);
    }

    /* CHAT SECTION */       
    h3.chat-section {
        font-size: 1.8rem;
        color: oklch(0.262 0.051 172.552);
        font-weight: bold;
        border-left: 6px solid oklch(0.596 0.145 163.225);
        padding-left: 12px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    
    @keyframes fadeIn { 
        from { opacity: 0; transform: translateY(20px); } 
        to { opacity: 1; transform: translateY(0); } 
    }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown(f"""
<div class="header-container">
    <div class="header-logo">
        <img src="{logo_src}" height="45">
        AeroPulse AI
    </div>
    <div style="display: flex; align-items: center; gap: 15px;">
        <span class="rag-badge">RAG ENABLED</span>
        <span class="rag-badge">v3.0</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ===== HERO SECTION =====
left_spacer, col_text, col_img, right_spacer = st.columns([0.5, 3, 3, 0.5], gap="large")

with col_text:
    st.write("")
    st.markdown(
        '<div class="hero-title">Predict Delays.<br>Travel Smarter.</div>', 
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-subtitle">Our Multi-Agent AI analyzes flights to give you risk assessments. Don\'t let a delay catch you off guard.</div>', 
        unsafe_allow_html=True
    )

with col_img:
    st.markdown(
        f'<img src="{hero_src}" style="width:100%; border-radius:20px; box-shadow: 0 20px 50px rgba(27,33,26,0.2);">', 
        unsafe_allow_html=True
    )

st.markdown("""
<div class="section-div">
    <span class="section-label">Start Your Analysis</span>
</div>
""", unsafe_allow_html=True)

# ===== INPUT FORM =====
input_c1, input_c2, input_c3 = st.columns([1, 2.5, 1]) 

with input_c2:

    with st.form(key="analysis_form"):
        if agent:
            c1, c2 = st.columns(2)
            airline = c1.selectbox("Airline", sorted(agent.get_airlines()))
            origin = c2.selectbox("Origin Airport", sorted(agent.get_origin()))
            
            c3, c4 = st.columns(2)
            dest = c3.selectbox("Destination Airport", sorted(agent.get_dest()))
            
            # Set minimum date to today
            date_val = c4.date_input(
                "Travel Date",
                min_value=datetime.today().date(),
                value=datetime.today().date(),
                max_value=datetime.today().date().replace(year=datetime.today().year + 1)
            )
            
            time_val = st.time_input(
                "Departure Time", 
                value=datetime.strptime("12:00", "%H:%M").time()
            )
        
        # Center the button logic inside the form
        predict_btn = st.form_submit_button(
            "Check Probability", 
            type="primary", 
            disabled=(not api_key)
        )

# ===== HANDLE ANALYSIS =====

    # Initialize session state variables if they don't exist
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    # Handle form submission
    if predict_btn and agent:
        # Validation 1: Check same origin and destination
        if origin == dest:
            st.error("❌ Origin and destination cannot be the same. Please choose different airports.")
            st.session_state.analysis_done = False
            st.session_state.analysis_result = None
        
        # Validation 2: Check if date is in the past
        elif date_val < datetime.today().date():
            st.error("❌ Travel date cannot be in the past. Please select today or a future date.")
            st.session_state.analysis_done = False
            st.session_state.analysis_result = None
        
        else:
            date_str = date_val.strftime("%Y-%m-%d")
            time_str = time_val.strftime("%H:%M")

            # Clear previous chat when new analysis runs
            st.session_state.messages = []
        
            with st.spinner("Retrieving data and generating advice..."):
                try:
                    result = agent.get_travel_advice(airline, origin, dest, date_str, time_str)
                
                    # Save to session state
                    st.session_state.analysis_result = result
                    st.session_state.analysis_done = True
                
                    # Store context for chat
                    st.session_state.context = {
                        "flight": f"{airline} from {origin} to {dest} on {date_str} at {time_str}",
                        "prediction": result['prediction'],
                        "confidence": result['confidence'],
                        "historical_reasons": result['historical_reasons'],
                        "rag_advice": result['rag_advice']
                    }
                
                except Exception as e:
                    st.error(f"Analysis Error: {e}")
                    st.session_state.analysis_done = False

# ===== RESULTS DISPLAY =====
    if st.session_state.analysis_done and st.session_state.analysis_result:
        result = st.session_state.analysis_result

    # Check for errors in result
        if "error" in result:
            st.error(f"{result['error']}")
        else:
            # Prepare content data
            rag_advice = result.get('rag_advice', {})
        
            # Determine Card Colors based on Risk
            if "HIGH" in result['prediction']:
                bg_color = "#FEF2F2"
                border = "#FCA5A5"
                text_c = "#991B1B"
                icon = "🛑"
                title = "High Risk of Delay"
            elif "MODERATE" in result['prediction']:
                bg_color = "#FFFBEB"
                border = "#FCD34D"
                text_c = "#92400E"
                icon = "⚠️"
                title = "Moderate Risk"
            else:
                bg_color = "#ECFDF5"
                border = "#6EE7B7"
                text_c = "#065F46"
                icon = "✅"
                title = "On Time Expected"

            # RISK CARD
            st.markdown(f"""
                <div class="result-card" style="background:{bg_color}; border:2px solid {border};">
                    <div style="font-size:4rem; margin-bottom: 10px;">{icon}</div>
                    <h1 style="margin: 10px 0; color: #1a202c;">{result['prediction']}</h1>
                    <p style="font-size: 1.1rem; color: #4a5568; margin: 5px 0;">
                        <b>Delay Probability:</b> {result['confidence']}
                    </p>
                </div>
            """, unsafe_allow_html=True)

# ===== RAG ANALYSIS =====
if st.session_state.analysis_done and st.session_state.analysis_result:
    result = st.session_state.analysis_result
    
    # Check for errors in result
    if "error" in result:
        st.error(f"{result['error']}")
    else:
        # Get rag_advice safely
        rag_advice = result.get('rag_advice', {})
        
        # ===== RAG ANALYSIS SECTION =====
        if not isinstance(rag_advice, dict) or 'error' in rag_advice:
            st.warning("Could not generate detailed advice. Check API configuration.")
        else:       
            # Risk Interpretation Text
            st.markdown(f"""
                <div class="rag-analysis">
                    <div style="font-size: 1.3rem; font-weight: 700; color: #1B211A;">
                        Detailed Analysis
                    </div>
                    <div style="font-size: 1rem; line-height: 1.7; color: #1B211A;">
                        {rag_advice.get('risk_interpretation', 'Analysis unavailable')}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # ===== KEY FACTORS =====
            st.markdown('<div class="section-title">Key Risk Factors</div>', unsafe_allow_html=True)
            for factor in rag_advice.get('key_factors', []):
                st.markdown(f'<span class="factor-badge">• {factor}</span>', unsafe_allow_html=True)

        # ===== CHAT INTERFACE =====
        st.markdown("<h3 class='chat-section'>Ask Follow-Up Questions</h3>", unsafe_allow_html=True)

        # Initialize chat if needed
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about alternatives, delay reasons, or travel tips..."):
            from agents import is_question_relevant
            
            # Check relevance
            if not is_question_relevant(prompt):
                with st.chat_message("assistant"):
                    st.markdown("Please ask questions related to flight delays, travel advice, or this analysis.")
            else:
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = agent.chat_with_user(
                            prompt,
                            st.session_state.messages[:-1], 
                            st.session_state.context
                        )
                        st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})

# ===== FOOTER =====
if "context" in st.session_state:
    st.markdown("""
        <div style="
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid oklch(0.596 0.145 163.225);
            text-align: center;
            color: #1B211A;
            opacity: 0.7;
            font-size: 0.85rem;
        ">
            <p>
                <b>Disclaimer:</b> Predictions are estimates based on historical data. 
                Always check with your airline for official flight status.
            </p>
        </div>
    """, unsafe_allow_html=True)