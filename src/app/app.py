"""Football Fit — entry point and navigation. Run from project root: streamlit run src/app/app.py"""
from pathlib import Path
import streamlit as st
from theme import inject_global_css

st.set_page_config(page_title="Football Fit | Tactical Recruitment Studio", page_icon="⚽", layout="wide")
inject_global_css()

logo_path = Path(__file__).parent / "assets" / "logo.svg"
if logo_path.exists():
    st.logo(str(logo_path), size="large")

navigation = st.navigation(
    {
        "Explore": [
            st.Page("views/0_Home.py", title="Overview", icon=":material/home:", default=True),
            st.Page("views/1_Player_Profile.py", title="Player Profile", icon=":material/person:"),
            st.Page("views/2_Club_Profile.py", title="Club Profile", icon=":material/shield:"),
        ],
        "Analyse": [
            st.Page("views/3_Fit_Analysis.py", title="Fit Analysis", icon=":material/compare_arrows:"),
            st.Page("views/4_Backtest_and_Style_Map.py", title="Style Map & Backtest", icon=":material/insights:"),
        ],
        "About": [
            st.Page("views/5_Methodology.py", title="Method & Limitations", icon=":material/menu_book:"),
        ],
    }
)
navigation.run()

#    cd D:\University\Project\football-fit
 #   .\.venv\Scripts\Activate.ps1
 #   streamlit run src\app\app.py