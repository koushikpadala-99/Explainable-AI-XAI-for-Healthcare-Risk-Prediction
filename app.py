import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global Reset & Typography */
*, *::before, *::after {
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
  box-sizing: border-box;
}

:root {
  --primary: #2563EB;
  --primary-hover: #1D4ED8;
  --success: #22C55E;
  --success-bg: #F0FDF4;
  --success-border: #BBF7D0;
  --danger: #EF4444;
  --danger-bg: #FEF2F2;
  --danger-border: #FECACA;
  --warning: #F59E0B;
  --warning-bg: #FFFBEB;
  --warning-border: #FDE68A;
  --bg-app: #F8FAFC;
  --border-color: #E2E8F0;
  --border-color-light: #F1F5F9;
  --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
  --card-radius: 14px;
}

#MainMenu, footer, header {
  visibility: hidden !important;
}

[data-testid="stDecoration"] {
  display: none !important;
}

@media (min-width: 1201px) {
  .stApp {
    background-color: var(--bg-app) !important;
    height: 100vh !important;
    overflow: hidden !important;
  }
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  .main,
  .block-container {
    height: 100vh !important;
    overflow: hidden !important;
  }
}

@media (max-width: 1200px) {
  .stApp {
    background-color: var(--bg-app) !important;
    height: auto !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
  }
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  .main,
  .block-container {
    height: auto !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
  }
}

/* Clear margins/paddings from standard containers */
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

[data-testid="stVerticalBlock"] {
  gap: 10px !important;
}

/* Style main content columns */
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"],
.block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] {
  padding: 12px 14px !important;
  align-items: stretch !important;
}

/* Left panel (Patient Input) Column */
@media (min-width: 1201px) {
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child,
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child,
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child,
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 24px -4px rgba(37,99,235,0.08), 0 1.5px 6px -1px rgba(0,0,0,0.04) !important;
    padding: 14px 14px 70px 14px !important;
    margin-top: 4px !important;
    flex: 0 0 330px !important;
    width: 330px !important;
    max-width: 330px !important;
    position: relative !important;
    overflow: hidden !important;
    height: calc(100vh - 28px) !important;
  }
}

@media (max-width: 1200px) {
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"],
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
    gap: 16px !important;
  }
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child,
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child,
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child,
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 24px -4px rgba(37,99,235,0.08), 0 1.5px 6px -1px rgba(0,0,0,0.04) !important;
    padding: 14px 14px 10px 14px !important;
    margin-top: 4px !important;
    flex: 1 1 100% !important;
    width: 100% !important;
    max-width: 100% !important;
    position: relative !important;
  }
}

/* Style the scrollable container inside patient input */
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child [data-testid="stVerticalBlockBorderWrapper"],
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stVerticalBlockBorderWrapper"],
.block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child [data-testid="stVerticalBlockBorderWrapper"],
.block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stVerticalBlockBorderWrapper"] {
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
}

/* Scrollbar inside the patient input container */
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div,
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div,
.block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div,
.block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div {
  scrollbar-width: thin !important;
  scrollbar-color: #93C5FD transparent !important;
}
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar,
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar,
.block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar,
.block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar {
  width: 5px !important;
}
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-thumb,
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-thumb,
.block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-thumb,
.block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-thumb {
  background: #93C5FD !important;
  border-radius: 999px !important;
}
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-track,
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-track,
.block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-track,
.block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-track {
  background: transparent !important;
}

@media (min-width: 1201px) {
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div,
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div,
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div,
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div {
    height: 100% !important;
    overflow-y: auto !important;
  }
}

@media (max-width: 1200px) {
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div,
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div,
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div,
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child [data-testid="stVerticalBlockBorderWrapper"] > div {
    height: 400px !important;
  }
}

/* Right panel Column */
@media (min-width: 1201px) {
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child,
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child,
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child,
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 4px 0 0 0 !important;
    flex: 1 !important;
    height: calc(100vh - 24px) !important;
    overflow-y: auto !important;
    scrollbar-width: thin;
    scrollbar-color: #CBD5E1 transparent;
  }
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child::-webkit-scrollbar,
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child::-webkit-scrollbar,
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child::-webkit-scrollbar,
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child::-webkit-scrollbar {
    width: 6px;
  }
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child::-webkit-scrollbar-thumb,
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child::-webkit-scrollbar-thumb,
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child::-webkit-scrollbar-thumb,
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child::-webkit-scrollbar-thumb {
    background: #CBD5E1;
    border-radius: 999px;
  }
}

@media (max-width: 1200px) {
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child,
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child,
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child,
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 10px 0 0 0 !important;
    flex: 1 1 100% !important;
    width: 100% !important;
    max-width: 100% !important;
    height: auto !important;
    overflow-y: visible !important;
  }
}

/* Custom vertical spacing inside the right column */
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child > [data-testid="stVerticalBlock"],
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"],
.block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child > [data-testid="stVerticalBlock"],
.block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] {
  gap: 12px !important;
}

/* All first-level columns inside right panel represent dashboard cards */
.block-container [data-testid="column"]:last-child [data-testid="stHorizontalBlock"] > [data-testid="column"],
.block-container [data-testid="stColumn"]:last-child [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
  background: #FFFFFF !important;
  border: 1px solid var(--border-color-light) !important;
  border-radius: var(--card-radius) !important;
  box-shadow: var(--card-shadow) !important;
  padding: 8px 12px !important;
  transition: all 0.3s ease !important;
}

.block-container [data-testid="column"]:last-child [data-testid="stHorizontalBlock"] > [data-testid="column"]:hover,
.block-container [data-testid="stColumn"]:last-child [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:hover {
  box-shadow: 0 8px 16px -2px rgba(0, 0, 0, 0.04), 0 4px 8px -2px rgba(0, 0, 0, 0.02) !important;
}

/* Ensure the top header row columns have transparent backgrounds */
.block-container [data-testid="column"]:last-child [data-testid="stHorizontalBlock"]:first-child > [data-testid="column"],
.block-container [data-testid="stColumn"]:last-child [data-testid="stHorizontalBlock"]:first-child > [data-testid="stColumn"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}

/* Style default widget labels in the left panel */
.block-container [data-testid="column"]:first-child label,
.block-container [data-testid="stColumn"]:first-child label,
.block-container [data-testid="column"]:first-child label *,
.block-container [data-testid="stColumn"]:first-child label *,
.block-container [data-testid="column"]:first-child [data-testid="stWidgetLabel"] p,
.block-container [data-testid="stColumn"]:first-child [data-testid="stWidgetLabel"] p {
  display: flex !important;
  align-items: center !important;
  font-size: 10.5px !important;
  font-weight: 600 !important;
  color: #475569 !important;
  -webkit-text-fill-color: #475569 !important;
  margin-bottom: 3px !important;
  margin-top: 4px !important;
}

/* Selectbox inputs styling */
.block-container [data-testid="column"]:first-child div[data-baseweb="select"],
.block-container [data-testid="stColumn"]:first-child div[data-baseweb="select"],
.block-container [data-testid="column"]:first-child div[data-baseweb="select"] > div,
.block-container [data-testid="stColumn"]:first-child div[data-baseweb="select"] > div {
  border-radius: 8px !important;
  border: 1px solid var(--border-color) !important;
  background-color: #FFFFFF !important;
  background: #FFFFFF !important;
  min-height: 30px !important;
  font-size: 11px !important;
  box-shadow: none !important;
  transition: all 0.2s ease !important;
}
.block-container [data-testid="column"]:first-child div[data-baseweb="select"]:hover,
.block-container [data-testid="stColumn"]:first-child div[data-baseweb="select"]:hover {
  border-color: #94A3B8 !important;
}
.block-container [data-testid="column"]:first-child div[data-baseweb="select"] span,
.block-container [data-testid="column"]:first-child div[data-baseweb="select"] div,
.block-container [data-testid="stColumn"]:first-child div[data-baseweb="select"] span,
.block-container [data-testid="stColumn"]:first-child div[data-baseweb="select"] div {
  color: #0F172A !important;
  -webkit-text-fill-color: #0F172A !important;
  font-size: 11px !important;
  background-color: transparent !important;
  background: transparent !important;
}
.block-container [data-testid="column"]:first-child div[data-baseweb="select"] svg,
.block-container [data-testid="stColumn"]:first-child div[data-baseweb="select"] svg {
  fill: #64748B !important;
  color: #64748B !important;
}

/* Number inputs styling */
.block-container [data-testid="column"]:first-child [data-testid="stNumberInput"] div[data-baseweb="input"],
.block-container [data-testid="stColumn"]:first-child [data-testid="stNumberInput"] div[data-baseweb="input"] {
  background-color: #FFFFFF !important;
  background: #FFFFFF !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 8px !important;
  min-height: 30px !important;
  box-shadow: none !important;
  transition: all 0.2s ease !important;
}
.block-container [data-testid="column"]:first-child [data-testid="stNumberInput"] div[data-baseweb="base-input"],
.block-container [data-testid="stColumn"]:first-child [data-testid="stNumberInput"] div[data-baseweb="base-input"] {
  background-color: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.block-container [data-testid="column"]:first-child [data-testid="stNumberInput"] div[data-baseweb="input"]:hover,
.block-container [data-testid="stColumn"]:first-child [data-testid="stNumberInput"] div[data-baseweb="input"]:hover {
  border-color: #94A3B8 !important;
}
.block-container [data-testid="column"]:first-child [data-testid="stNumberInput"] input,
.block-container [data-testid="stColumn"]:first-child [data-testid="stNumberInput"] input {
  color: #0F172A !important;
  -webkit-text-fill-color: #0F172A !important;
  background-color: transparent !important;
  font-size: 11px !important;
  padding: 4px 8px !important;
  border: none !important;
}

/* Number input step buttons */
.block-container [data-testid="column"]:first-child [data-testid="stNumberInput"] button,
.block-container [data-testid="stColumn"]:first-child [data-testid="stNumberInput"] button {
  background-color: #F8FAFC !important;
  background: #F8FAFC !important;
  color: #64748B !important;
  border: none !important;
  border-left: 1px solid var(--border-color) !important;
  min-height: 28px !important;
  border-radius: 0 !important;
}
.block-container [data-testid="column"]:first-child [data-testid="stNumberInput"] button:hover,
.block-container [data-testid="stColumn"]:first-child [data-testid="stNumberInput"] button:hover {
  background-color: #F1F5F9 !important;
  color: #0F172A !important;
}
.block-container [data-testid="column"]:first-child [data-testid="stNumberInput"] button svg,
.block-container [data-testid="stColumn"]:first-child [data-testid="stNumberInput"] button svg {
  fill: #64748B !important;
  color: #64748B !important;
}

/* Dropdown list popover styling */
div[data-baseweb="popover"] {
  background-color: #FFFFFF !important;
  background: #FFFFFF !important;
  border-radius: 8px !important;
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
  border: 1px solid var(--border-color) !important;
}
div[data-baseweb="popover"] ul[role="listbox"] {
  background-color: #FFFFFF !important;
  background: #FFFFFF !important;
  padding: 4px !important;
}
div[data-baseweb="popover"] li[role="option"] {
  background-color: #FFFFFF !important;
  background: #FFFFFF !important;
  color: #1E293B !important;
  font-size: 11px !important;
  padding: 6px 10px !important;
  border-radius: 6px !important;
  margin-bottom: 2px !important;
  transition: all 0.15s ease !important;
}
div[data-baseweb="popover"] li[role="option"]:hover,
div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {
  background-color: #F1F5F9 !important;
  background: #F1F5F9 !important;
  color: #0F172A !important;
}


/* Predict button — absolutely pinned to bottom of the left panel */
@media (min-width: 1201px) {
  .block-container [data-testid="column"]:first-child div[data-testid="stButton"],
  .block-container [data-testid="stColumn"]:first-child div[data-testid="stButton"] {
    position: absolute !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    background: #FFFFFF !important;
    padding: 10px 14px 12px 14px !important;
    border-top: 1px solid var(--border-color-light) !important;
    z-index: 10 !important;
    width: 100% !important;
    box-sizing: border-box !important;
  }
}

/* On mobile, just render normally */
@media (max-width: 1200px) {
  .block-container [data-testid="column"]:first-child div[data-testid="stButton"],
  .block-container [data-testid="stColumn"]:first-child div[data-testid="stButton"] {
    background: #FFFFFF !important;
    padding: 10px 0px 6px 0px !important;
    border-top: 1px solid var(--border-color-light) !important;
    width: 100% !important;
  }
}


.block-container [data-testid="column"]:first-child div[data-testid="stButton"] > button,
.block-container [data-testid="stColumn"]:first-child div[data-testid="stButton"] > button {
  width: 100% !important;
  background: var(--primary) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 6px 12px !important;
  font-size: 12px !important;
  font-weight: 700 !important;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
  transition: all 0.2s ease !important;
  margin-top: 2px !important;
}
.block-container [data-testid="column"]:first-child div[data-testid="stButton"] > button:hover,
.block-container [data-testid="stColumn"]:first-child div[data-testid="stButton"] > button:hover {
  background: var(--primary-hover) !important;
  box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3) !important;
  transform: translateY(-1px) !important;
}
.block-container [data-testid="column"]:first-child div[data-testid="stButton"] > button:active,
.block-container [data-testid="stColumn"]:first-child div[data-testid="stButton"] > button:active {
  transform: translateY(0px) !important;
}

/* Hide Streamlit Plotly padding */
[data-testid="stPlotlyChart"] > div {
  margin: 0 !important;
  padding: 0 !important;
}
[data-testid="stPlotlyChart"] {
  width: 100% !important;
  max-width: 100% !important;
  overflow: hidden !important;
}

/* Factor list styles */
ul.risk-list {
  list-style: none;
  padding: 0;
  margin: 0;
}
ul.risk-list li {
  padding-left: 12px;
  position: relative;
  margin-bottom: 0px;
  font-size: 9px;
  color: #475569;
  line-height: 1.2;
}
ul.risk-list li::before {
  content: "•";
  color: var(--danger);
  font-weight: bold;
  display: inline-block;
  width: 14px;
  margin-left: -14px;
}

ul.safe-list {
  list-style: none;
  padding: 0;
  margin: 0;
}
ul.safe-list li {
  padding-left: 12px;
  position: relative;
  margin-bottom: 0px;
  font-size: 9px;
  color: #475569;
  line-height: 1.2;
}
ul.safe-list li::before {
  content: "•";
  color: var(--success);
  font-weight: bold;
  display: inline-block;
  width: 14px;
  margin-left: -14px;
}

@media (max-width: 1200px) {
  .block-container [data-testid="column"]:last-child [data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
    gap: 12px !important;
  }
  .block-container [data-testid="column"]:last-child [data-testid="stHorizontalBlock"] > [data-testid="column"],
  .block-container [data-testid="stColumn"]:last-child [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    flex: 1 1 100% !important;
    width: 100% !important;
    max-width: 100% !important;
  }
}

@media (max-width: 992px) {
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"],
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
    gap: 16px !important;
  }
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"],
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="column"],
  .block-container > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    width: 100% !important;
    max-width: 100% !important;
    flex: 1 1 100% !important;
  }
}

@media (max-width: 576px) {
  .block-container [data-testid="column"]:last-child [data-testid="stHorizontalBlock"]:first-child {
    flex-direction: column !important;
    gap: 8px !important;
  }
  .block-container [data-testid="column"]:last-child [data-testid="stHorizontalBlock"]:first-child > [data-testid="column"],
  .block-container [data-testid="stColumn"]:last-child [data-testid="stHorizontalBlock"]:first-child > [data-testid="stColumn"] {
    flex: 1 1 100% !important;
    width: 100% !important;
    max-width: 100% !important;
  }
}

@media (min-width: 577px) and (max-width: 1200px) {
  .block-container [data-testid="column"]:last-child [data-testid="stHorizontalBlock"]:first-child {
    flex-direction: row !important;
  }
  .block-container [data-testid="column"]:last-child [data-testid="stHorizontalBlock"]:first-child > [data-testid="column"],
  .block-container [data-testid="stColumn"]:last-child [data-testid="stHorizontalBlock"]:first-child > [data-testid="stColumn"] {
    width: auto !important;
    max-width: none !important;
    flex: 1 !important;
  }
}
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_explainer():
    with open("heart_disease_prediction_model.pkl", "rb") as f:
        mdl = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        fnames = pickle.load(f)
    
    # Generate mock background for LIME with realistic UCI distribution
    np.random.seed(42)
    n_samples = 150
    mock_data = {
        "age": np.random.normal(54, 9, n_samples).clip(29, 77).astype(int),
        "sex": np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
        "chest_pain_type": np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.5, 0.2, 0.2, 0.1]),
        "resting_blood_pressure": np.random.normal(131, 17, n_samples).clip(94, 200).astype(int),
        "cholesterol": np.random.normal(246, 51, n_samples).clip(126, 564).astype(int),
        "fbs": np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
        "restecg": np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.48, 0.02]),
        "max_heart_rate": np.random.normal(149, 22, n_samples).clip(71, 202).astype(int),
        "exercise_induced_angina": np.random.choice([0, 1], size=n_samples, p=[0.67, 0.33]),
        "oldpeak": np.random.exponential(1.0, n_samples).clip(0, 6.2),
        "slope": np.random.choice([0, 1, 2], size=n_samples, p=[0.46, 0.46, 0.08]),
        "major_vessels": np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.58, 0.21, 0.13, 0.08]),
        "thalassemia": np.random.choice([1, 2, 3], size=n_samples, p=[0.06, 0.54, 0.4])
    }
    bg = pd.DataFrame(mock_data, columns=fnames)
    
    import lime.lime_tabular
    explainer = lime.lime_tabular.LimeTabularExplainer(
        bg.values,
        feature_names=fnames,
        class_names=["No Disease", "Disease"],
        mode="classification",
        discretize_continuous=True
    )
    return mdl, fnames, explainer

model, feature_names, lime_explainer = load_model_and_explainer()

FEATURE_LABELS = {
    "age":"Age","sex":"Sex","chest_pain_type":"Chest Pain",
    "resting_blood_pressure":"Resting BP","cholesterol":"Cholesterol",
    "fbs":"Blood Sugar","restecg":"Resting ECG",
    "max_heart_rate":"Max Heart Rate","exercise_induced_angina":"Exer. Angina",
    "oldpeak":"ST Depression","slope":"ST Slope",
    "major_vessels":"Major Vessels","thalassemia":"Thalassemia",
}
# ── Layout columns ──────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 3], gap="small")

# ── LEFT PANEL ──────────────────────────────────────────────────────────────────
with left_col:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid #F1F5F9;">
      <span style="display:flex;align-items:center;justify-content:center;color:#2563EB;flex-shrink:0;">
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="8.5" cy="7" r="4"></circle><line x1="20" y1="8" x2="20" y2="14"></line><line x1="23" y1="11" x2="17" y2="11"></line></svg>
      </span>
      <div>
        <div style="font-size:14px;font-weight:800;color:#2563EB;line-height:1.2;letter-spacing:-0.3px;">Patient Input</div>
        <div style="font-size:10px;color:#64748B;margin-top:1px;font-weight:500;">Enter details to predict risk</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Scrollable container for all patient inputs ────────────────────────────
    input_container = st.container(height=int(500), border=False)
    with input_container:
        cp_map      = {"Typical Angina":0,"Atypical Angina":1,"Non-Anginal Pain":2,"Asymptomatic":3}
        exang_map   = {"No":0,"Yes":1}
        fbs_map     = {"No (≤120 mg/dl)":0,"Yes (>120 mg/dl)":1}
        restecg_map = {"Normal":0,"ST-T Wave Abnormality":1,"LV Hypertrophy":2}
        slope_map   = {"Downsloping":0,"Flat":1,"Upsloping":2}
        thal_map    = {"Normal":1,"Fixed Defect":2,"Reversible Defect":3}

        # Stacking all inputs vertically in logical order
        age = st.number_input("📅 Age (years)", min_value=1, max_value=120, value=50, step=1)
        sex = st.selectbox("👤 Sex", ["Male","Female"])
        cp = st.selectbox("❤️ Chest Pain Type", list(cp_map.keys()))
        trestbps = st.number_input("🩸 Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120, step=1)
        chol = st.number_input("🧪 Cholesterol (mg/dl)", min_value=50, max_value=600, value=200, step=1)
        fbs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl", list(fbs_map.keys()))
        restecg = st.selectbox("⚡ Resting ECG", list(restecg_map.keys()))
        thalach = st.number_input("💓 Max Heart Rate Achieved", min_value=50, max_value=250, value=150, step=1)
        exang = st.selectbox("🏃‍♂️ Exercise Induced Angina", list(exang_map.keys()))
        oldpeak = st.number_input("📉 ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
        slope = st.selectbox("📈 Slope of ST", list(slope_map.keys()))
        ca = st.selectbox("🚢 Number of Major Vessels (0-3)", [0,1,2,3])
        thal = st.selectbox("🧬 Thalassemia", list(thal_map.keys()))

    predict_clicked = st.button("🩺  Predict Risk", use_container_width=True)
    st.markdown("""<div style="font-size:9px;color:#94A3B8;text-align:center;margin-top:4px;line-height:1.3;">
      ⚕️ Educational purposes only. Not a medical advice substitute.</div>""", unsafe_allow_html=True)

    # Track which button was last pressed in session state
    if predict_clicked:
        st.session_state["active_view"] = "full"

# ── Chart helpers ───────────────────────────────────────────────────────────────
# ── Chart helpers ───────────────────────────────────────────────────────────────
def get_gradient_color(val):
    # val is from 0 to 100
    # Interpolate between green (#22C55E), yellow (#EAB308), orange (#F97316), red (#EF4444)
    # green: (34, 197, 94)
    # yellow: (234, 179, 8)
    # orange: (249, 115, 22)
    # red: (239, 68, 68)
    t = val / 100.0
    if t < 0.33:
        factor = t / 0.33
        r = int(34 + (234 - 34) * factor)
        g = int(197 + (179 - 197) * factor)
        b = int(94 + (8 - 94) * factor)
    elif t < 0.66:
        factor = (t - 0.33) / 0.33
        r = int(234 + (249 - 234) * factor)
        g = int(179 + (115 - 179) * factor)
        b = int(8 + (22 - 8) * factor)
    else:
        factor = (t - 0.66) / 0.34
        r = int(249 + (239 - 249) * factor)
        g = int(115 + (68 - 115) * factor)
        b = int(22 + (68 - 22) * factor)
    return f"rgb({r},{g},{b})"

def gauge_chart(prob):
    # 1. Donut slices for arc
    values = [1]*100 + [100]
    colors = [get_gradient_color(i) for i in range(100)] + ["rgba(0,0,0,0)"]
    
    pie_trace = go.Pie(
        values=values,
        labels=[f"{i}%" for i in range(101)] + ["bottom"],
        marker=dict(colors=colors),
        hole=0.70,
        rotation=180,
        sort=False,
        direction="clockwise",
        hoverinfo="none",
        showlegend=False,
        textinfo="none"
    )
    
    # 2. Needle line
    phi = np.pi * (1.0 - prob / 100.0)
    r_needle = 0.65
    needle_x = [0, r_needle * np.cos(phi)]
    needle_y = [0, r_needle * np.sin(phi)]
    
    needle_trace = go.Scatter(
        x=needle_x,
        y=needle_y,
        mode="lines",
        line=dict(color="#0F172A", width=3.5),
        hoverinfo="none",
        showlegend=False
    )
    
    # 3. Center cap
    center_cap = go.Scatter(
        x=[0],
        y=[0],
        mode="markers",
        marker=dict(
            size=14,
            color="#0F172A",
            line=dict(color="#FFFFFF", width=2)
        ),
        hoverinfo="none",
        showlegend=False
    )
    
    fig = go.Figure(data=[pie_trace, needle_trace, center_cap])
    
    clr_text = "#EF4444" if prob >= 70 else ("#F59E0B" if prob >= 40 else "#22C55E")
    
    # Text annotations in polar center
    fig.add_annotation(
        x=0, y=0.25,
        text=f"<b>{prob:.2f}%</b>",
        font=dict(size=20, color=clr_text, family="Plus Jakarta Sans", weight="bold"),
        showarrow=False,
        yref="y", xref="x"
    )
    fig.add_annotation(
        x=0, y=-0.3,
        text="Risk of Heart Disease",
        font=dict(size=8.5, color="#64748B", family="Plus Jakarta Sans", weight="bold"),
        showarrow=False,
        yref="y", xref="x"
    )
    
    # Outer bottom labels
    fig.add_annotation(
        x=-0.80, y=-0.15,
        text="<b>0%</b>",
        font=dict(size=8, color="#64748B", family="Plus Jakarta Sans"),
        showarrow=False,
        yref="y", xref="x"
    )
    fig.add_annotation(
        x=0.80, y=-0.15,
        text="<b>100%</b>",
        font=dict(size=8, color="#64748B", family="Plus Jakarta Sans"),
        showarrow=False,
        yref="y", xref="x"
    )
    
    fig.update_layout(
        height=140,
        margin=dict(t=2, b=2, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.1, 1.1]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.1, 1.1],
            scaleanchor="x",
            scaleratio=1
        )
    )
    return fig

def shap_bar(shap_vals, feat_names, top_n=8):
    sv = shap_vals[0] if len(shap_vals.shape)>1 else shap_vals
    pairs = sorted(zip(feat_names,sv), key=lambda x:abs(x[1]), reverse=True)[:top_n]
    labels = [FEATURE_LABELS.get(f,f) for f,_ in pairs]
    values = [v for _,v in pairs]
    colors = ["#EF4444" if v>0 else "#3B82F6" for v in values]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
         marker_color=colors, marker_line_width=0,
         text=[f"{v:+.2f}" for v in values], textposition="outside",
         textfont={"size":8.5,"color":"#475569", "family": "Plus Jakarta Sans"},
         hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>"))
    fig.update_layout(height=140, margin=dict(t=15,b=10,l=2,r=40),
         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
         xaxis=dict(tickfont={"size":8},zeroline=True,zerolinecolor="#E2E8F0",
                    zerolinewidth=2,gridcolor="#F8FAFC"),
         yaxis=dict(autorange="reversed",tickfont={"size":8.5,"color":"#374151"}),
         font={"family":"Plus Jakarta Sans"},showlegend=False)
    return fig, labels, values

def lime_bar(lime_list, top_n=8):
    pairs = sorted(lime_list, key=lambda x:abs(x[1]), reverse=True)[:top_n]
    raw, values = [p[0] for p in pairs], [p[1] for p in pairs]
    clean = []
    for lbl in raw:
        matched = lbl
        for k,v in FEATURE_LABELS.items():
            if k in lbl.lower(): matched = v; break
        clean.append(matched)
    colors = ["#22C55E" if v>0 else "#EF4444" for v in values]
    fig = go.Figure(go.Bar(x=values, y=clean, orientation="h",
         marker_color=colors, marker_line_width=0,
         text=[f"{v:.2f}" for v in values], textposition="outside",
         textfont={"size":8.5,"color":"#475569", "family": "Plus Jakarta Sans"},
         hovertemplate="<b>%{y}</b><br>LIME: %{x:.4f}<extra></extra>"))
    fig.update_layout(height=140, margin=dict(t=15,b=10,l=2,r=40),
         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
         xaxis=dict(tickfont={"size":8},zeroline=True,zerolinecolor="#E2E8F0",
                    zerolinewidth=2,gridcolor="#F8FAFC"),
         yaxis=dict(autorange="reversed",tickfont={"size":8.5,"color":"#374151"}),
         font={"family":"Plus Jakarta Sans"},showlegend=False)
    return fig, clean, values

def radar_chart(sl, sv, ll, lv, top_n=8):
    common = sl[:top_n]
    sn = [abs(v) for v in sv[:top_n]]; mx = max(sn+[1e-9]); sn = [v/mx for v in sn]
    ld = dict(zip(ll,[abs(v) for v in lv])); lmx = max(list(ld.values())+[1e-9])
    ln = [ld.get(l,0)/lmx for l in common]
    theta = common + [common[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=sn+[sn[0]],theta=theta,fill="toself",name="SHAP",
        line=dict(color="#3B82F6",width=2),fillcolor="rgba(59,130,246,0.12)"))
    fig.add_trace(go.Scatterpolar(r=ln+[ln[0]],theta=theta,fill="toself",name="LIME",
        line=dict(color="#EF4444",width=2),fillcolor="rgba(239,68,68,0.12)"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True,range=[0,1],tickfont={"size":7,"color":"#475569"},
                                   gridcolor="#E2E8F0",linecolor="#E2E8F0"),
                   angularaxis=dict(tickfont={"size":8.5,"color":"#374151"})),
        showlegend=True,
        legend=dict(orientation="h",y=-0.25,x=0.5,xanchor="center",yanchor="top",
                    font={"size":8.5,"color":"#374151"},bgcolor="rgba(0,0,0,0)"),
        height=140, margin=dict(t=10,b=25,l=35,r=35),
        paper_bgcolor="rgba(0,0,0,0)", font={"family":"Plus Jakarta Sans", "color":"#374151"})
    return fig

# ── RIGHT PANEL ─────────────────────────────────────────────────────────────────
with right_col:

    # ── Top Header inside Right Column ──────────────────────────────────────────
    col_title, col_badge1, col_badge2 = st.columns([3.2, 0.9, 0.9], gap="small")
    with col_title:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom: 6px;">
          <div style="width:34px; height:34px; border-radius:50%;
                      background:#2563EB;
                      display:flex; align-items:center; justify-content:center;
                      color:#FFFFFF; box-shadow:0 4px 12px rgba(37,99,235,0.25);">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
          </div>
          <div>
            <div style="font-size:18px; font-weight:800; color:#0F172A; letter-spacing:-0.4px; line-height:1.2;">
              Heart Disease Risk Prediction
            </div>
            <div style="font-size:10px; color:#64748B; font-weight:500; margin-top:2px; display:flex; align-items:center; gap:4px;">
              <span>AI Powered</span><span style="color:#CBD5E1;">•</span><span>Explainable</span><span style="color:#CBD5E1;">•</span><span>Reliable</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_badge1:
        st.markdown("""
        <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px; padding:3px 8px; display:flex; flex-direction:column; justify-content:center; height:34px; margin-bottom: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
          <div style="font-size:7px; color:#2563EB; font-weight:700; text-transform:uppercase; letter-spacing:.08em; display:flex; align-items:center; gap:2px;">
            <span style="width:2.5px; height:2.5px; border-radius:50%; background:#2563EB; display:inline-block;"></span>Model
          </div>
          <div style="font-size:9.5px; font-weight:700; color:#1E3A8A; display:flex; align-items:center; gap:3px;">
            ⚙️ XGBoost
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_badge2:
        st.markdown("""
        <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:3px 8px; display:flex; flex-direction:column; justify-content:center; height:34px; margin-bottom: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
          <div style="font-size:7px; color:#16A34A; font-weight:700; text-transform:uppercase; letter-spacing:.08em; display:flex; align-items:center; gap:2px;">
            <span style="width:2.5px; height:2.5px; border-radius:50%; background:#16A34A; display:inline-block;"></span>Status
          </div>
          <div style="font-size:9.5px; font-weight:700; color:#14532D; display:flex; align-items:center; gap:3px;">
            <span style="width:4px; height:4px; border-radius:50%; background:#22C55E; box-shadow:0 0 0 2px rgba(34,197,94,0.15); display:inline-block;"></span>
            Loaded
          </div>
        </div>
        """, unsafe_allow_html=True)


    # ── Idle state — show welcome message until a button is pressed ─────────
    active_view = st.session_state.get("active_view", None)

    if active_view is None:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    min-height:340px;gap:14px;padding:32px 16px;text-align:center;">
          <div style="width:64px;height:64px;border-radius:50%;background:#EFF6FF;
                      display:flex;align-items:center;justify-content:center;
                      box-shadow:0 4px 16px rgba(37,99,235,0.12);">
            <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="#2563EB"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
          </div>
          <div>
            <div style="font-size:16px;font-weight:800;color:#1E293B;margin-bottom:6px;">
              Ready to Analyze
            </div>
            <div style="font-size:11px;color:#64748B;line-height:1.6;max-width:340px;font-weight:500;">
              Fill in the patient details in the left panel, then click
              <strong style="color:#2563EB;">🩺 Predict Risk</strong> to view the full dashboard.
            </div>
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-top:4px;">
            <span style="font-size:9.5px;font-weight:600;color:#64748B;background:#F8FAFC;
                         border:1px solid #E2E8F0;border-radius:6px;padding:4px 10px;">
              ⚙️ XGBoost Model
            </span>
            <span style="font-size:9.5px;font-weight:600;color:#64748B;background:#F8FAFC;
                         border:1px solid #E2E8F0;border-radius:6px;padding:4px 10px;">
              🔍 SHAP + LIME Explainability
            </span>
            <span style="font-size:9.5px;font-weight:600;color:#64748B;background:#F8FAFC;
                         border:1px solid #E2E8F0;border-radius:6px;padding:4px 10px;">
              📊 Risk Visualization
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Build input & run prediction ──────────────────────────────────────
        vals = {
            "age": age, "sex": 1 if sex=="Male" else 0,
            "chest_pain_type": cp_map[cp],
            "resting_blood_pressure": trestbps,
            "cholesterol": chol, "fbs": fbs_map[fbs],
            "restecg": restecg_map[restecg], "max_heart_rate": thalach,
            "exercise_induced_angina": exang_map[exang],
            "oldpeak": oldpeak, "slope": slope_map[slope],
            "major_vessels": ca, "thalassemia": thal_map[thal],
        }
        input_df    = pd.DataFrame([[vals[f] for f in feature_names]], columns=feature_names)
        prediction  = model.predict(input_df)[0]
        proba       = model.predict_proba(input_df)[0]
        prob_d      = float(proba[1]) * 100
        prob_nd     = float(proba[0]) * 100

        is_high = prob_d >= 70; is_med = 40 <= prob_d < 70
        r_lbl   = "HIGH RISK" if is_high else ("MODERATE RISK" if is_med else "LOW RISK")
        r_clr   = "#DC2626"   if is_high else ("#F59E0B" if is_med else "#16A34A")
        r_bg    = "#FEF2F2"   if is_high else ("#FFFBEB" if is_med else "#F0FDF4")
        r_brd   = "#FECACA"   if is_high else ("#FDE68A" if is_med else "#BBF7D0")
        r_icon  = "🔴" if is_high else ("🟡" if is_med else "🟢")
        conf_lbl = "High" if abs(prob_d-50)>30 else "Moderate"
        conf_clr = "#16A34A" if conf_lbl=="High" else "#F59E0B"

        shap_fig = lime_fig = radar_fig = None
        shap_labels = shap_values_row = lime_labels = lime_values = []

        try:
            import shap
            explainer = shap.TreeExplainer(model)
            sv        = explainer.shap_values(input_df)
            shap_fig, shap_labels, shap_values_row = shap_bar(sv, feature_names)
        except Exception:
            pass

        try:
            lr = lime_explainer.explain_instance(
                input_df.values[0],
                model.predict_proba,
                num_features=10,
                top_labels=1
            )
            lkey = list(lr.local_exp.keys())[0]
            ll = [(feature_names[i], w) for i, w in lr.local_exp[lkey]]
            lime_fig, lime_labels, lime_values = lime_bar(ll)
        except Exception:
            pass

        if shap_fig and lime_fig:
            try:
                radar_fig = radar_chart(shap_labels, shap_values_row, lime_labels, lime_values)
            except Exception:
                pass

        def draw_svg_gauge(prob, clr_text):
            import math
            theta = math.pi * (1.0 - prob / 100.0)
            r_needle = 50
            x_end = 80 + r_needle * math.cos(theta)
            y_end = 75 - r_needle * math.sin(theta)
            svg = f"""
            <svg viewBox="0 0 160 88" style="width: 100%; max-height: 110px; overflow: visible;">
              <defs>
                <linearGradient id="gauge-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="#22C55E" />
                  <stop offset="40%" stop-color="#EAB308" />
                  <stop offset="70%" stop-color="#F97316" />
                  <stop offset="100%" stop-color="#EF4444" />
                </linearGradient>
              </defs>
              <path d="M 20 75 A 60 60 0 0 1 140 75" fill="none" stroke="#F1F5F9" stroke-width="9" stroke-linecap="round" />
              <path d="M 20 75 A 60 60 0 0 1 140 75" fill="none" stroke="url(#gauge-grad)" stroke-width="9" stroke-linecap="round" />
              <line x1="80" y1="75" x2="{x_end}" y2="{y_end}" stroke="#0F172A" stroke-width="3" stroke-linecap="round" />
              <circle cx="80" cy="75" r="4.5" fill="#0F172A" stroke="#FFFFFF" stroke-width="1.5" />
              <text x="80" y="54" text-anchor="middle" font-family="Plus Jakarta Sans, sans-serif" font-weight="800" font-size="16" fill="{clr_text}">{prob:.1f}%</text>
              <text x="80" y="66" text-anchor="middle" font-family="Plus Jakarta Sans, sans-serif" font-weight="700" font-size="7" fill="#64748B">Risk of Heart Disease</text>
              <text x="20" y="86" text-anchor="middle" font-family="Plus Jakarta Sans, sans-serif" font-weight="700" font-size="6.5" fill="#94A3B8">0%</text>
              <text x="140" y="86" text-anchor="middle" font-family="Plus Jakarta Sans, sans-serif" font-weight="700" font-size="6.5" fill="#94A3B8">100%</text>
            </svg>
            """
            return svg

        # ── PREDICT RESULTS view: focused summary card ───────────────────────
        if active_view == "results":
            import streamlit.components.v1 as components
            gauge_svg_html = draw_svg_gauge(prob_d, r_clr)
            pred_txt = ("Likely to have heart disease." if prediction == 1
                        else "Unlikely to have heart disease.")
            pred_badge = (
                f'<span style="font-size:11px;font-weight:700;color:#EF4444;'
                f'background:#FEF2F2;border:1px solid #FECACA;border-radius:6px;padding:3px 10px;">1 — Disease</span>'
                if prediction == 1 else
                f'<span style="font-size:11px;font-weight:700;color:#22C55E;'
                f'background:#F0FDF4;border:1px solid #BBF7D0;border-radius:6px;padding:3px 10px;">0 — No Disease</span>'
            )
            conf_badge = (
                f'<span style="font-size:11px;font-weight:700;color:#22C55E;background:#F0FDF4;'
                f'border:1px solid #BBF7D0;border-radius:6px;padding:3px 10px;">High</span>'
                if conf_lbl == "High" else
                f'<span style="font-size:11px;font-weight:700;color:#F59E0B;background:#FFFBEB;'
                f'border:1px solid #FDE68A;border-radius:6px;padding:3px 10px;">Moderate</span>'
            )
            results_html = f"""
            <!DOCTYPE html><html><head><style>
              body {{ margin:0; padding:0; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; background:transparent; }}
              .results-card {{ background:#FFFFFF; border:1.5px solid #E2E8F0; border-radius:16px;
                               padding:20px 24px; box-shadow:0 4px 24px -4px rgba(37,99,235,0.10); }}
              .results-title {{ font-size:13px; font-weight:800; color:#1E293B; margin-bottom:16px;
                                display:flex; align-items:center; gap:8px; border-bottom:1px solid #F1F5F9; padding-bottom:10px; }}
              .gauge-wrap {{ display:flex; justify-content:center; margin-bottom:16px; }}
              .row {{ display:flex; justify-content:space-between; align-items:center;
                      padding:8px 0; border-bottom:1px solid #F8FAFC; }}
              .row:last-child {{ border-bottom:none; }}
              .row-label {{ font-size:10px; font-weight:600; color:#64748B; text-transform:uppercase; letter-spacing:0.05em; }}
              .risk-pill {{ display:inline-block; background:{r_bg}; border:1.5px solid {r_brd};
                            color:{r_clr}; border-radius:999px; padding:4px 14px;
                            font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:0.05em; }}
              .disclaimer {{ font-size:8.5px; color:#94A3B8; text-align:center; margin-top:14px; font-weight:500; }}
            </style></head><body>
              <div class="results-card">
                <div class="results-title">
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#2563EB" stroke-width="2.5"
                       stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                  </svg>
                  Prediction Results
                </div>
                <div class="gauge-wrap">{gauge_svg_html}</div>
                <div style="text-align:center; margin-bottom:14px;">
                  <div class="risk-pill">{r_icon}&nbsp; {r_lbl}</div>
                  <div style="font-size:9.5px;color:#64748B;margin-top:6px;font-weight:500;">{pred_txt}</div>
                </div>
                <div class="row">
                  <span class="row-label">Prediction</span>
                  {pred_badge}
                </div>
                <div class="row">
                  <span class="row-label">Disease Probability</span>
                  <span style="font-size:13px;font-weight:800;color:{r_clr};">{prob_d:.1f}%</span>
                </div>
                <div class="row">
                  <span class="row-label">No-Disease Probability</span>
                  <span style="font-size:13px;font-weight:800;color:#22C55E;">{prob_nd:.1f}%</span>
                </div>
                <div class="row">
                  <span class="row-label">Confidence</span>
                  {conf_badge}
                </div>
                <div class="row">
                  <span class="row-label">Model</span>
                  <span style="font-size:11px;font-weight:700;color:#2563EB;background:#EFF6FF;
                               border:1px solid #BFDBFE;border-radius:6px;padding:3px 10px;">⚙️ XGBoost</span>
                </div>
                <div class="disclaimer">⚕️ For educational purposes only. Not a substitute for medical advice.</div>
              </div>
            </body></html>
            """
            components.html(results_html, height=500, scrolling=False)

        # ── PREDICT RISK view: full dashboard ────────────────────────────────
        else:
            # ── ROW 1 ────────────────────────────────────────────────────────
            c1, c2 = st.columns([1.8, 1.0], gap="small")

            with c1:
                st.markdown("""
                <div style="margin-bottom: 2px;">
                  <div style="font-size: 11px; font-weight: 700; color: #1E293B; display: flex; align-items: center; gap: 4px;">
                    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
                    Risk Overview
                  </div>
                  <div style="font-size: 9px; color: #64748B; margin-left: 17px; margin-top: 1px;">Probability of Heart Disease</div>
                </div>
                """, unsafe_allow_html=True)
                
                pred_txt = ("Likely to have heart disease."
                            if prediction==1 else
                            "Unlikely to have heart disease.")
                
                import streamlit.components.v1 as components
                
                gauge_svg_html = draw_svg_gauge(prob_d, r_clr)
                
                # Determine HTML badges for Prediction, Risk Level, Confidence
                pred_badge = f'<span style="font-size: 8.5px; font-weight: 700; color: #EF4444; background: #FEF2F2; border: 1px solid #FECACA; border-radius: 6px; padding: 2.5px 8px; text-transform: uppercase;">1 (Disease)</span>' if prediction==1 else f'<span style="font-size: 8.5px; font-weight: 700; color: #22C55E; background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 2.5px 8px; text-transform: uppercase;">0 (No Disease)</span>'
                risk_badge = f'<span style="font-size: 8.5px; font-weight: 700; color: {r_clr}; background: {r_bg}; border: 1px solid {r_brd}; border-radius: 6px; padding: 2.5px 8px; text-transform: uppercase;">{r_lbl}</span>'
                conf_badge = f'<span style="font-size: 8.5px; font-weight: 700; color: #22C55E; background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 2.5px 8px; text-transform: uppercase;">{conf_lbl}</span>' if conf_lbl=="High" else f'<span style="font-size: 8.5px; font-weight: 700; color: #F59E0B; background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 6px; padding: 2.5px 8px; text-transform: uppercase;">{conf_lbl}</span>'

                html_code = f"""
                <!DOCTYPE html>
                <html>
                <head>
                  <style>
                    body {{
                      margin: 0;
                      padding: 0;
                      background-color: transparent;
                      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                      overflow: hidden;
                    }}
                    .card-container {{
                      display: flex;
                      flex-direction: row;
                      align-items: center;
                      gap: 14px;
                      width: 100%;
                      height: 110px;
                      box-sizing: border-box;
                    }}
                    .left-side {{
                      flex: 1.1;
                      display: flex;
                      flex-direction: column;
                      align-items: center;
                      justify-content: center;
                      text-align: center;
                    }}
                    .gauge-container {{
                      width: 100%;
                      max-width: 140px;
                      margin-bottom: 2px;
                    }}
                    .status-pill-container {{
                      margin-top: -6px;
                      margin-bottom: 4px;
                    }}
                    .status-pill {{
                      display: inline-block;
                      background: {r_bg};
                      border: 1.5px solid {r_brd};
                      color: {r_clr};
                      border-radius: 999px;
                      padding: 2px 10px;
                      font-size: 8.5px;
                      font-weight: 700;
                      text-transform: uppercase;
                      letter-spacing: 0.05em;
                    }}
                    .status-sub {{
                      font-size: 8px;
                      color: #64748B;
                      margin-top: 2px;
                      font-weight: 500;
                    }}
                    .divider {{
                      width: 1px;
                      background: #F1F5F9;
                      align-self: stretch;
                      margin: 4px 0;
                    }}
                    .right-side {{
                      flex: 1.0;
                      display: flex;
                      flex-direction: column;
                      justify-content: center;
                      gap: 5px;
                      padding-left: 4px;
                    }}
                    .detail-box {{
                      background: #FAFBFC;
                      border: 1px solid #E2E8F0;
                      border-radius: 8px;
                      padding: 5px 12px;
                      display: flex;
                      justify-content: space-between;
                      align-items: center;
                      box-shadow: 0 1px 2px rgba(0,0,0,0.01);
                      box-sizing: border-box;
                    }}
                    .detail-label {{
                      font-size: 8.5px;
                      color: #64748B;
                      font-weight: 600;
                      text-transform: uppercase;
                      letter-spacing: 0.05em;
                    }}
                  </style>
                </head>
                <body>
                  <div class="card-container">
                    <div class="left-side">
                      <div class="gauge-container">
                        {gauge_svg_html}
                      </div>
                      <div class="status-pill-container">
                        <div class="status-pill">{r_lbl}</div>
                        <div class="status-sub">{pred_txt}</div>
                      </div>
                    </div>
                    
                    <div class="divider"></div>
                    
                    <div class="right-side">
                      <div class="detail-box">
                        <span class="detail-label">Prediction</span>
                        {pred_badge}
                      </div>
                      <div class="detail-box">
                        <span class="detail-label">Risk Level</span>
                        {risk_badge}
                      </div>
                      <div class="detail-box">
                        <span class="detail-label">Confidence</span>
                        {conf_badge}
                      </div>
                    </div>
                  </div>
                </body>
                </html>
                """
                components.html(html_code, height=115)

            with c2:
                st.markdown(f"""
                <div style="margin-bottom: 4px;">
                  <div style="font-size: 11px; font-weight: 700; color: #2563EB; display: flex; align-items: center; gap: 6px;">
                    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                    Prediction Summary
                  </div>
                </div>
                """, unsafe_allow_html=True)
                
                icon_target = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>'
                icon_shield = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="m9 11 2 2 4-4"></path></svg>'
                icon_gear = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>'
                icon_chart = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>'
                icon_star = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>'
                
                def summary_row(icon_svg, label, value, last=False):
                    border_style = "" if last else "border-bottom: 1px solid #F1F5F9;"
                    return f"""
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 2px 0; {border_style}">
                      <div style="display: flex; align-items: center; gap: 6px;">
                        <div style="width: 18px; height: 18px; border-radius: 5px; background: #EFF6FF; color: #2563EB; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                          {icon_svg}
                        </div>
                        <span style="font-size: 9.5px; color: #475569; font-weight: 600;">{label}</span>
                      </div>
                      {value}
                    </div>
                    """
                
                # Compute dynamic badges for Prediction Summary values
                p_dis_clr = "#EF4444" if prob_d >= 70 else ("#F59E0B" if prob_d >= 40 else "#22C55E")
                p_dis_bg = "#FEF2F2" if prob_d >= 70 else ("#FFFBEB" if prob_d >= 40 else "#F0FDF4")
                p_dis_brd = "#FECACA" if prob_d >= 70 else ("#FDE68A" if prob_d >= 40 else "#BBF7D0")
                p_dis_badge = f'<span style="font-size: 8.5px; font-weight: 700; color: {p_dis_clr}; background: {p_dis_bg}; border: 1px solid {p_dis_brd}; border-radius: 5px; padding: 2px 7px;">{prob_d:.1f}%</span>'

                p_ndis_clr = "#22C55E" if prob_nd >= 70 else ("#F59E0B" if prob_nd >= 40 else "#EF4444")
                p_ndis_bg = "#F0FDF4" if prob_nd >= 70 else ("#FFFBEB" if prob_nd >= 40 else "#FEF2F2")
                p_ndis_brd = "#BBF7D0" if prob_nd >= 70 else ("#FDE68A" if prob_nd >= 40 else "#FECACA")
                p_ndis_badge = f'<span style="font-size: 8.5px; font-weight: 700; color: {p_ndis_clr}; background: {p_ndis_bg}; border: 1px solid {p_ndis_brd}; border-radius: 5px; padding: 2px 7px;">{prob_nd:.1f}%</span>'

                model_badge = f'<span style="font-size: 8.5px; font-weight: 700; color: #2563EB; background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 5px; padding: 2px 7px;">XGBoost</span>'
                acc_badge = f'<span style="font-size: 8.5px; font-weight: 700; color: #475569; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 5px; padding: 2px 7px;">~98.53%</span>'
                conf_summary_badge = f'<span style="font-size: 8.5px; font-weight: 700; color: #22C55E; background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 5px; padding: 2px 7px;">High</span>' if conf_lbl=="High" else f'<span style="font-size: 8.5px; font-weight: 700; color: #F59E0B; background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 5px; padding: 2px 7px;">Moderate</span>'

                rows_html = (
                    summary_row(icon_target, "Prob. (Disease)", p_dis_badge) +
                    summary_row(icon_shield, "Prob. (No Disease)", p_ndis_badge) +
                    summary_row(icon_gear, "Model Used", model_badge) +
                    summary_row(icon_chart, "Model Accuracy", acc_badge) +
                    summary_row(icon_star, "Confidence", conf_summary_badge, last=True)
                )
                st.markdown(rows_html, unsafe_allow_html=True)

            # ── ROW 2: Explainability ─────────────────────────────────────────────────
            st.markdown("""
            <div style="font-size: 12.5px; font-weight: 800; color: #2563EB; margin-top: 12px; margin-bottom: 8px; padding-left: 6px; border-left: 3px solid #2563EB; line-height: 1.1;">
              Explainability Dashboard
            </div>""", unsafe_allow_html=True)

            e1, e2, e3 = st.columns([1.0, 1.0, 1.0], gap="small")

            with e1:
                st.markdown("""
                <div style="padding-bottom: 4px; border-bottom: 1px solid #F1F5F9; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                  <div>
                    <div style="font-size: 11px; font-weight: 700; color: #1E293B;">SHAP Explanation</div>
                    <div style="font-size: 8.5px; color: #64748B; margin-top: 1px;">Top Features (Impact on Prediction)</div>
                  </div>
                  <div style="color: #94A3B8; cursor: pointer;" title="SHAP (SHapley Additive exPlanations) shows how each feature contributes to the prediction compared to the average model prediction.">
                    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                  </div>
                </div>""", unsafe_allow_html=True)
                if shap_fig:
                    st.plotly_chart(shap_fig, use_container_width=True,
                                    config={"displayModeBar": False})
                    st.markdown("""
                    <div style="display:flex;gap:12px;justify-content:center;font-size:9.5px;
                                color:#64748B;padding-bottom:0px;margin-top:2px;">
                      <span style="display:flex;align-items:center;gap:3px;">
                        <i style="width:8px;height:8px;background:#EF4444;border-radius:2px;display:inline-block;"></i>
                        Increase Risk</span>
                      <span style="display:flex;align-items:center;gap:3px;">
                        <i style="width:8px;height:8px;background:#3B82F6;border-radius:2px;display:inline-block;"></i>
                        Decrease Risk</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.info("SHAP data unavailable.")

            with e2:
                st.markdown("""
                <div style="padding-bottom: 4px; border-bottom: 1px solid #F1F5F9; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                  <div>
                    <div style="font-size: 11px; font-weight: 700; color: #1E293B;">LIME Explanation</div>
                    <div style="font-size: 8.5px; color: #64748B; margin-top: 1px;">Local Feature Importance (LIME)</div>
                  </div>
                  <div style="color: #94A3B8; cursor: pointer;" title="LIME (Local Interpretable Model-agnostic Explanations) builds a local surrogate model around the patient's data point to explain the decision.">
                    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                  </div>
                </div>""", unsafe_allow_html=True)
                if lime_fig:
                    st.plotly_chart(lime_fig, use_container_width=True,
                                    config={"displayModeBar": False})
                    st.markdown("""
                    <div style="display:flex;gap:12px;justify-content:center;font-size:9.5px;
                                color:#64748B;padding-bottom:0px;margin-top:2px;">
                      <span style="display:flex;align-items:center;gap:3px;">
                        <i style="width:8px;height:8px;background:#22C55E;border-radius:2px;display:inline-block;"></i>
                        Positive Impact</span>
                      <span style="display:flex;align-items:center;gap:3px;">
                        <i style="width:8px;height:8px;background:#EF4444;border-radius:2px;display:inline-block;"></i>
                        Negative Impact</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.info("LIME data unavailable.")

            with e3:
                st.markdown("""
                <div style="padding-bottom: 4px; border-bottom: 1px solid #F1F5F9; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                  <div>
                    <div style="font-size: 11px; font-weight: 700; color: #1E293B;">SHAP vs LIME Comparison</div>
                    <div style="font-size: 8.5px; color: #64748B; margin-top: 1px;">Feature Importance Comparison</div>
                  </div>
                  <div style="color: #94A3B8; cursor: pointer;" title="Comparison of feature importance values normalized between SHAP and LIME.">
                    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                  </div>
                </div>""", unsafe_allow_html=True)
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True,
                                    config={"displayModeBar": False})
                else:
                    st.info("Radar comparison unavailable.")

            # ── ROW 3: Feature Impact + Recommendation ────────────────────────────────
            risk_factors, safe_factors = [], []
            if shap_labels and shap_values_row:
                for lbl, v in zip(shap_labels, shap_values_row):
                    if v > 0.005:   risk_factors.append(lbl)
                    elif v < -0.005: safe_factors.append(lbl)
            else:
                if oldpeak > 1.5:                   risk_factors.append("oldpeak")
                if chol > 240:                      risk_factors.append("cholesterol")
                if exang == "Yes":                  risk_factors.append("exercise_induced_angina")
                if cp == "Asymptomatic":            risk_factors.append("chest_pain_type")
                if thalach > 140:                   safe_factors.append("max_heart_rate")
                if restecg == "Normal":             safe_factors.append("restecg")
                if ca == 0:                         safe_factors.append("major_vessels")
                if exang == "No":                   safe_factors.append("exercise_induced_angina")

            def get_factor_desc(feature_name):
                fname = str(feature_name).lower()
                if "age" in fname:
                    return f"Age ({age} years)"
                elif "sex" in fname:
                    return "Male Patient" if sex == "Male" else "Female Patient"
                elif "chest" in fname or "cp" in fname:
                    return f"Chest Pain: {cp}"
                elif "blood" in fname or "bp" in fname or "bps" in fname or "trest" in fname:
                    return f"Resting BP ({trestbps} mm Hg)"
                elif "chol" in fname:
                    return f"Cholesterol ({chol} mg/dl)"
                elif "fbs" in fname or "sugar" in fname:
                    return "Fasting Blood Sugar > 120 mg/dl" if fbs.startswith("Yes") else "Normal Blood Sugar"
                elif "ecg" in fname or "restecg" in fname:
                    return f"Resting ECG: {restecg}"
                elif "heart" in fname or "rate" in fname or "thalach" in fname:
                    return f"Max Heart Rate ({thalach} bpm)"
                elif "angina" in fname or "exang" in fname:
                    return "Exercise Induced Angina" if exang == "Yes" else "No Exercise Induced Angina"
                elif "oldpeak" in fname or "depression" in fname:
                    return f"ST Depression ({oldpeak})"
                elif "slope" in fname:
                    return f"ST Slope: {slope}"
                elif "vessel" in fname or "ca" in fname:
                    return f"Major Vessels Blocked: {ca}"
                elif "thal" in fname:
                    return f"Thalassemia: {thal}"
                return str(feature_name)

            risk_bullets = "".join([f'<li>{get_factor_desc(f)}</li>' for f in risk_factors[:4]]) or \
                           '<li>No significant risk-increasing factors</li>'
            safe_bullets = "".join([f'<li>{get_factor_desc(f)}</li>' for f in safe_factors[:4]]) or \
                           '<li>No significant protective factors detected</li>'

            # Recommendation text
            if prob_d >= 70:
                r_card_bg, r_card_brd = "#FEF2F2", "#FECACA"
                rx_title = "Immediate Medical Attention Required"
                rx_short = "Consult a cardiologist immediately for diagnostic evaluation and necessary medical tests."
            elif prob_d >= 40:
                r_card_bg, r_card_brd = "#FFFBEB", "#FDE68A"
                rx_title = "Further Medical Evaluation Recommended"
                rx_short = "Further evaluation is recommended — consider a lipid profile, stress test, and lifestyle counseling."
            else:
                r_card_bg, r_card_brd = "#F0FDF4", "#BBF7D0"
                rx_title = "Low Risk — Continue Preventive Care"
                rx_short = "Continue healthy lifestyle: regular exercise, balanced diet, and routine annual check-ups."

            # Base64 encode heart image
            import base64
            import os
            from PIL import Image
            from io import BytesIO
            heart_b64 = ""
            if os.path.exists("heart_indicator.png"):
                try:
                    with Image.open("heart_indicator.png") as img:
                        img.thumbnail((100, 100))  # Resize to max 100x100 for crisp display
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        heart_b64 = base64.b64encode(buffered.getvalue()).decode()
                except Exception:
                    try:
                        with open("heart_indicator.png", "rb") as f:
                            heart_b64 = base64.b64encode(f.read()).decode()
                    except Exception:
                        pass

            fi1, fi2 = st.columns([1.6, 1.2], gap="small")

            with fi1:
                st.markdown(f"""
                <div style="margin-bottom: 4px;">
                  <div style="font-size: 11px; font-weight: 700; color: #2563EB; display: flex; align-items: center; gap: 4px;">
                    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>
                    Feature Impact Summary
                  </div>
                </div>
                
                <div style="display: flex; gap: 10px; margin-top: 6px; flex-wrap: wrap; min-height: 108px;">
                  <!-- Risk Increasing Factors -->
                  <div style="flex: 1; min-width: 190px; display: flex; gap: 8px; align-items: flex-start; padding: 8px 12px; border: 1px solid #FECACA; border-radius: 10px; background: #FEF2F2; min-height: 108px; height: auto; box-sizing: border-box; box-shadow: 0 1px 2px rgba(0,0,0,0.01);">
                    <div style="width: 20px; height: 20px; border-radius: 50%; background: #EF4444; color: #FFFFFF; display: flex; align-items: center; justify-content: center; flex-shrink: 0; box-shadow: 0 2px 4px rgba(239,68,68,0.2);">
                      <svg viewBox="0 0 24 24" width="10" height="10" fill="none" stroke="currentColor" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>
                    </div>
                    <div>
                      <div style="font-size: 10.5px; font-weight: 700; color: #991B1B; margin-bottom: 2px;">Risk Increasing Factors</div>
                      <ul class="risk-list">
                        {risk_bullets}
                      </ul>
                    </div>
                  </div>
                  
                  <!-- Risk Decreasing Factors -->
                  <div style="flex: 1; min-width: 190px; display: flex; gap: 8px; align-items: flex-start; padding: 8px 12px; border: 1px solid #BBF7D0; border-radius: 10px; background: #F0FDF4; min-height: 108px; height: auto; box-sizing: border-box; box-shadow: 0 1px 2px rgba(0,0,0,0.01);">
                    <div style="width: 20px; height: 20px; border-radius: 50%; background: #22C55E; color: #FFFFFF; display: flex; align-items: center; justify-content: center; flex-shrink: 0; box-shadow: 0 2px 4px rgba(34,197,94,0.2);">
                      <svg viewBox="0 0 24 24" width="10" height="10" fill="none" stroke="currentColor" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><polyline points="19 12 12 19 5 12"></polyline></svg>
                    </div>
                    <div>
                      <div style="font-size: 10.5px; font-weight: 700; color: #166534; margin-bottom: 2px;">Risk Decreasing Factors</div>
                      <ul class="safe-list">
                        {safe_bullets}
                      </ul>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with fi2:
                recommendation_card_html = f"""<div style="display: flex; flex-direction: column; justify-content: space-between; min-height: 108px; height: 100%;">
                <div style="margin-bottom: 4px;">
                  <div style="font-size: 11px; font-weight: 700; color: #2563EB; display: flex; align-items: center; gap: 6px;">
                    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                    Recommendation
                  </div>
                </div>
                <div style="display: flex; gap: 10px; align-items: center; background: {r_card_bg}; border: 1px solid {r_card_brd}; border-radius: 10px; padding: 10px 12px; min-height: 108px; height: auto; box-sizing: border-box; box-shadow: 0 1px 2px rgba(0,0,0,0.01);">
                  <div style="flex: 1;">
                    <div style="font-size: 10px; font-weight: 800; color: #1E293B; margin-bottom: 3px;">{rx_title}</div>
                    <div style="font-size: 8.5px; color: #475569; line-height: 1.35; font-weight: 500;">
                      This patient is at <strong>{r_lbl}</strong>.<br>
                      {rx_short}
                    </div>
                  </div>"""
                if heart_b64:
                    recommendation_card_html += f"""
                  <div style="width: 50px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                    <img src="data:image/png;base64,{heart_b64}" style="width: 44px; height: 44px; object-fit: contain; filter: drop-shadow(0 4px 8px rgba(220,38,38,0.1));" />
                  </div>"""
                recommendation_card_html += """</div></div>"""
                st.markdown(recommendation_card_html, unsafe_allow_html=True)

            # ── Footer ───────────────────────────────────────────────────────────────────
            st.markdown("""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 6px 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-top: 8px; box-shadow: var(--card-shadow);">
              <div style="display: flex; align-items: center; gap: 6px; color: #64748B; font-size: 9.5px; font-weight: 500;">
                <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: #94A3B8;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
              </div>
              <div style="display: flex; align-items: center; gap: 4px; color: #64748B; font-size: 9.5px; font-weight: 600;">
                <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: #94A3B8;"><path d="M12 22c5.523 0 10-2.239 10-5V5c0-2.761-4.477-5-10-5S2 2.239 2 5v12c0 2.761 4.477 5 10 5z"></path><path d="M2 5c0 2.761 4.477 5 10 5s10-2.239 10-5"></path><path d="M2 11c0 2.761 4.477 5 10 5s10-2.239 10-5"></path></svg>
                UCI Heart Disease Dataset
              </div>
            </div>
            """, unsafe_allow_html=True)
