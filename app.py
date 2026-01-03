import streamlit as st
import sys
import os

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="HAR Activity Recognition Toolkit",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar
)

# -------------------------------------------------
# CUSTOM CSS (HIDE SIDEBAR, STYLE HEADER)
# -------------------------------------------------
st.markdown("""
<style>
    /* Hide the default sidebar */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 2.5rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 0 0 20px 20px;
        margin: -1rem -1rem 3rem -1rem;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .main-header h1 {
        font-size: 3.2rem;
        margin-bottom: 0.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .main-header h3 {
        font-size: 1.4rem;
        font-weight: 400;
        opacity: 0.95;
        margin-bottom: 1.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.85;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Navigation bar styling */
    .nav-bar {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0 3rem 0;
        flex-wrap: wrap;
    }
    
    .nav-button {
        padding: 0.8rem 1.8rem;
        border-radius: 12px;
        background: white;
        color: #764ba2;
        border: 2px solid #764ba2;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
    }
    
    .nav-button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .nav-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
    }
    
    /* Module cards */
    .module-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .module-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid #eaeaea;
        transition: all 0.4s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        position: relative;
        overflow: hidden;
    }
    
    .module-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.12);
    }
    
    .module-card:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .card-icon {
        font-size: 3.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .card-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .card-description {
        color: #5d6d7e;
        line-height: 1.7;
        margin-bottom: 1.5rem;
        flex-grow: 1;
        font-size: 1.05rem;
    }
    
    .card-team {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        font-size: 0.95rem;
        color: #6c757d;
    }
    
    .team-label {
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.3rem;
    }
    
    /* Acknowledgements section */
    .ack-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin: 4rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .ack-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .ack-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .ack-card {
        background: rgba(255, 255, 255, 0.15);
        padding: 1.5rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .ack-card-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: white;
    }
    
    .ack-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .ack-list li {
        margin-bottom: 0.8rem;
        padding-left: 1.5rem;
        position: relative;
    }
    
    .ack-list li:before {
        content: "•";
        position: absolute;
        left: 0;
        color: white;
        font-size: 1.2rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        border-top: 1px solid #eaeaea;
        margin-top: 3rem;
    }
    
    .footer-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #495057;
    }
    
    /* Back button for module pages */
    .back-button {
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 1000;
        background: white;
        border: 2px solid #667eea;
        color: #667eea;
        padding: 0.7rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .back-button:hover {
        background: #667eea;
        color: white;
        transform: translateX(-5px);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.2rem;
        }
        
        .module-grid {
            grid-template-columns: 1fr;
        }
        
        .nav-bar {
            flex-direction: column;
            align-items: center;
        }
        
        .nav-button {
            width: 100%;
            max-width: 300px;
        }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# PATH FIX
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# -------------------------------------------------
# IMPORT MODULES
# -------------------------------------------------
def safe_import(module_name):
    """Safely import a module"""
    try:
        module = __import__(module_name)
        return module, True
    except ImportError:
        return None, False

swimming_module, swimming_available = safe_import("swimming")
walk_module, walk_available = safe_import("walk_toerise_run")
custom_module, custom_available = safe_import("customactivity")

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
def render_home():
    """Render the home page with no sidebar"""
    
    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1>🏋️‍♂️ Human Activity Recognition Toolkit</h1>
        <h3>Intelligent Systems – Group Work Assignment</h3>
        <p>
            Advanced analysis of physical activities using smartphone and wearable sensor data.
            Developed by students at NOVIA University of Applied Sciences.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Bar
    current_page = st.session_state.page
    st.markdown('<div class="nav-bar">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        is_active = "active" if current_page == "home" else ""
        if st.button("🏠 **Home**", use_container_width=True, key="nav_home"):
            st.session_state.page = "home"
            st.rerun()
    
    with col2:
        is_active = "active" if current_page == "swimming" else ""
        if st.button("🏊‍♂️ **Swimming**", use_container_width=True, key="nav_swim"):
            st.session_state.page = "swimming"
            st.rerun()
    
    with col3:
        is_active = "active" if current_page == "walk" else ""
        if st.button("🚶‍♂️ **Walk/Run**", use_container_width=True, key="nav_walk"):
            st.session_state.page = "walk"
            st.rerun()
    
    with col4:
        is_active = "active" if current_page == "custom" else ""
        if st.button("⚙️ **Custom**", use_container_width=True, key="nav_custom"):
            st.session_state.page = "custom"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Acknowledgements Section
    st.markdown("""
    <div class="ack-section">
        <div class="ack-title">🎓 Acknowledgements & Credits</div>
        
        <div class="ack-grid">
            <div class="ack-card">
                <div class="ack-card-title">Course Leadership</div>
                <ul class="ack-list">
                    <li><strong>Ray</strong> - Course Leader & Lecturer</li>
                    <li>Intelligent Systems Course</li>
                    <li>NOVIA University of Applied Sciences</li>
                </ul>
            </div>
            
            <div class="ack-card">
                <div class="ack-card-title">Swimming Analysis Team</div>
                <ul class="ack-list">
                    <li><strong>Ellen Williamsson</strong> - Elite Swimmer</li>
                    <li>Medley stroke expert</li>
                    <li>Apple Watch IMU data</li>
                </ul>
            </div>
            
            <div class="ack-card">
                <div class="ack-card-title">Walk/Run Analysis Team</div>
                <ul class="ack-list">
                    <li><strong>Suresh Lama</strong> - Lead Developer</li>
                    <li><strong>Joans Williamsson</strong> - Data Analyst</li>
                    <li><strong>Ramandeep Singh</strong> - ML Engineer</li>
                </ul>
            </div>
            
            <div class="ack-card">
                <div class="ack-card-title">Custom Activity Team</div>
                <ul class="ack-list">
                    <li>Open to all students</li>
                    <li>Community contributions</li>
                    <li>Educational research focus</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Module Cards
    st.markdown("<h2 style='text-align: center; margin: 2rem 0 3rem 0; color: #2c3e50;'>🔍 Select Analysis Tool</h2>", unsafe_allow_html=True)
    
    st.markdown('<div class="module-grid">', unsafe_allow_html=True)
    
    # Swimming Card
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div class="module-card">
            <div class="card-icon">🏊‍♂️</div>
            <div class="card-title">Swimming Stroke Recognition</div>
            <div class="card-description">
                Advanced analysis of medley swimming strokes using Apple Watch IMU data.
                Classify butterfly, backstroke, breaststroke, and freestyle with 16 specialized features.
                Perfect for swim coaches and performance analysis.
            </div>
            <div class="card-team">
                <div class="team-label">Team & Data:</div>
                Ellen Williamsson (Elite Swimmer) • Apple Watch Sensors • 100Hz IMU Data
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if swimming_available:
            if st.button("**Launch** 🚀", key="launch_swim", use_container_width=True, type="primary"):
                st.session_state.page = "swimming"
                st.rerun()
        else:
            st.warning("⚠️ Not available")
    
    # Walk/Run Card
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div class="module-card">
            <div class="card-icon">🚶‍♂️</div>
            <div class="card-title">Walk/Run/Toe-Rise Analysis</div>
            <div class="card-description">
                Comprehensive gait analysis using smartphone accelerometer data.
                Detect walking, running, toe-rising, and knee-bending activities.
                Ideal for rehabilitation monitoring and daily activity tracking.
            </div>
            <div class="card-team">
                <div class="team-label">Team:</div>
                Suresh Lama • Joans Williamsson • Ramandeep Singh • Smartphone Sensors
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if walk_available:
            if st.button("**Launch** 🚀", key="launch_walk", use_container_width=True, type="primary"):
                st.session_state.page = "walk"
                st.rerun()
        else:
            st.warning("⚠️ Not available")
    
    # Custom Card
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div class="module-card">
            <div class="card-icon">⚙️</div>
            <div class="card-title">Custom Activity Creator</div>
            <div class="card-description">
                Create personalized activity recognition models.
                Define custom activities, set time ranges, and train ML models for any movement.
                Perfect for research projects and new activity discovery.
            </div>
            <div class="card-team">
                <div class="team-label">Open Community:</div>
                All students welcome • Educational use • Research projects • Custom development
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if custom_available:
            if st.button("**Launch** 🚀", key="launch_custom", use_container_width=True, type="primary"):
                st.session_state.page = "custom"
                st.rerun()
        else:
            st.warning("⚠️ Not available")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Technical Info
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 About This Assignment")
        st.markdown("""
        This toolkit implements Human Activity Recognition (HAR) concepts from the **Intelligent Systems** course.
        
        **Educational Objectives:**
        - Practical ML application
        - Sensor data processing
        - Feature engineering
        - Real-world problem solving
        
        **Tools:** Python, Streamlit, Scikit-learn, Plotly
        """)
    
    with col2:
        st.markdown("### 🔧 Technical Specifications")
        st.markdown("""
        **Data Requirements:**
        - CSV format (Phyphox export)
        - 100Hz sampling rate
        - Accelerometer + Gyroscope
        - Timestamped data
        
        **ML Pipeline:**
        - Data segmentation
        - Feature extraction
        - KNN & Random Forest
        - Performance evaluation
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <div class="footer-title">🎓 NOVIA University of Applied Sciences</div>
        <p>Intelligent Systems Course • Group Work Assignment • HAR Toolkit v1.0</p>
        <p style="font-size: 0.9rem; margin-top: 1rem; opacity: 0.7;">
            For educational purposes • © 2024
        </p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# MODULE PAGES (No sidebar, just back button)
# -------------------------------------------------
def render_module_page(module, module_name, display_name):
    """Render a module page with no sidebar"""
    
    # Back button (fixed position)
    st.markdown("""
    <style>
    .back-button-container {
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 1000;
    }
    </style>
    <div class="back-button-container">
    """, unsafe_allow_html=True)
    
    if st.button("← **Back to Home**", key=f"back_{module_name}"):
        st.session_state.page = "home"
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Module content
    if module:
        try:
            if hasattr(module, 'render'):
                # Call the module's render function
                module.render()
            else:
                st.error(f"Module {module_name}.py needs a 'render()' function!")
                st.code("""
                # Add this to your {module_name}.py:
                def render():
                    # Your Streamlit code here
                    st.title("Your Module")
                """)
                
                if st.button("Return to Home", key=f"return_{module_name}"):
                    st.session_state.page = "home"
                    st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
            if st.button("← Return to Home", key=f"error_back_{module_name}"):
                st.session_state.page = "home"
                st.rerun()
    else:
        st.error(f"Module '{module_name}' not found!")
        st.info(f"Create a file named '{module_name}.py' in the same directory")
        
        if st.button("← Return to Home", key=f"missing_back_{module_name}"):
            st.session_state.page = "home"
            st.rerun()

# -------------------------------------------------
# MAIN ROUTER
# -------------------------------------------------
def main():
    """Main router - no sidebar code here"""
    
    # Route to appropriate page
    if st.session_state.page == "home":
        render_home()
    elif st.session_state.page == "swimming":
        render_module_page(swimming_module, "swimming", "🏊‍♂️ Swimming Analysis")
    elif st.session_state.page == "walk":
        render_module_page(walk_module, "walk", "🚶‍♂️ Walk/Run Analysis")
    elif st.session_state.page == "custom":
        render_module_page(custom_module, "custom", "⚙️ Custom Activity Creator")
    else:
        st.session_state.page = "home"
        st.rerun()

# -------------------------------------------------
# RUN THE APP
# -------------------------------------------------
if __name__ == "__main__":
    main()