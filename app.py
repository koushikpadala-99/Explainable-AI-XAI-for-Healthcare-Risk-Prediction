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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*,*::before,*::after{font-family:'Inter',sans-serif!important;box-sizing:border-box;}
#MainMenu,footer,header{visibility:hidden!important;}
[data-testid="stDecoration"]{display:none!important;}
.stApp{background:#F8FAFC!important;}
.block-container{padding:0!important;max-width:100%!important;}
[data-testid="stVerticalBlock"],[data-testid="stVerticalBlockBorderWrapper"]{gap:0!important;}
.element-container{margin:0!important;padding:0!important;}
[data-testid="stHorizontalBlock"]{gap:0!important;align-items:flex-start!important;}
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{padding:0!important;}
@media(min-width:992px){
  [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:first-child{
    position:sticky!important;top:0!important;height:100vh!important;
    overflow-y:auto!important;scrollbar-width:thin;scrollbar-color:#CBD5E1 transparent;
  }
}
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:first-child::-webkit-scrollbar{width:4px;}
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:first-child::-webkit-scrollbar-thumb{background:#CBD5E1;border-radius:999px;}
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:first-child{
  background:#FFFFFF!important;border-right:1px solid #E2E8F0!important;
  box-shadow:4px 0 20px rgba(15,23,42,0.06)!important;
}
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:last-child{
  padding:18px 20px 32px!important;overflow-x:hidden!important;
}
/* Kill element-container spacing inside right panel */
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:last-child .element-container{
  margin:0!important;padding:0!important;
}
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:last-child [data-testid="stVerticalBlock"]{
  gap:0!important;
}
/* Collapse gap between card-title block and chart block inside same column */
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > .element-container + .element-container{
  margin-top:0!important;
}
[data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] > div{
  gap:0!important;
}
/* Plotly chart iframe wrapper — no extra spacing */
[data-testid="stPlotlyChart"] > div { margin:0!important; padding:0!important; }
.stSelectbox label,.stNumberInput label{
  font-size:11px!important;font-weight:600!important;
  color:#374151!important;margin-bottom:2px!important;
}
div[data-baseweb="select"]>div{
  border-radius:10px!important;border:1.5px solid #E2E8F0!important;
  min-height:40px!important;font-size:13px!important;background:#FAFBFC!important;
}
div[data-baseweb="select"]>div:focus-within{border-color:#2563EB!important;}
div[data-baseweb="input"] input{
  border-radius:10px!important;border:1.5px solid #E2E8F0!important;
  min-height:40px!important;font-size:13px!important;background:#FAFBFC!important;
}
div[data-baseweb="input"] input:focus{border-color:#2563EB!important;outline:none!important;}
div[data-testid="stButton"]>button{
  width:100%!important;
  background:linear-gradient(135deg,#DC2626 0%,#B91C1C 100%)!important;
  color:#fff!important;border:none!important;border-radius:12px!important;
  padding:14px 0!important;font-size:14px!important;font-weight:700!important;
  box-shadow:0 8px 24px rgba(220,38,38,0.30)!important;
  transition:all .2s ease!important;
}
div[data-testid="stButton"]>button:hover{
  background:linear-gradient(135deg,#B91C1C 0%,#991B1B 100%)!important;
  transform:translateY(-1px)!important;
}
.stPlotlyChart,[data-testid="stPlotlyChart"]{
  border-radius:10px!important;overflow:hidden!important;
  width:100%!important;margin:0!important;padding:0!important;
}
.card{
  background:#FFFFFF;border:1px solid #E2E8F0;border-radius:16px;
  padding:18px 20px;box-shadow:0 2px 12px rgba(15,23,42,0.05);margin-bottom:14px;
}
.ctitle{font-size:13px;font-weight:700;color:#1E293B;margin-bottom:3px;}
.csub{font-size:11px;color:#94A3B8;margin-bottom:10px;}
.srow{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #F1F5F9;}
.srow:last-child{border-bottom:none;padding-bottom:0;}
.slbl{font-size:12px;color:#64748B;font-weight:500;}
.pill{border-radius:999px;padding:5px 11px;font-size:11px;font-weight:600;display:inline-block;margin:2px 2px 2px 0;}
.pr{background:#FEF2F2;color:#DC2626;border:1px solid #FECACA;}
.pg{background:#F0FDF4;color:#16A34A;border:1px solid #BBF7D0;}
.rec-h{background:linear-gradient(135deg,#FFF1F2,#FEE2E2);border-left:4px solid #DC2626;border-radius:0 12px 12px 0;padding:14px 16px;}
.rec-m{background:linear-gradient(135deg,#FFFBEB,#FEF3C7);border-left:4px solid #F59E0B;border-radius:0 12px 12px 0;padding:14px 16px;}
.rec-l{background:linear-gradient(135deg,#F0FDF4,#DCFCE7);border-left:4px solid #16A34A;border-radius:0 12px 12px 0;padding:14px 16px;}
.grp{font-size:10px;font-weight:700;color:#94A3B8;letter-spacing:.1em;text-transform:uppercase;margin:14px 0 5px 0;}
.secbar{display:flex;align-items:center;gap:8px;margin:18px 0 10px;}
.secbar-line{width:3px;height:18px;border-radius:999px;}
.secbar-txt{font-size:14px;font-weight:800;color:#1E293B;}

/* Right panel only: let nested column blocks size to content to avoid large empty gaps */
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:last-child [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]>[data-testid="stVerticalBlock"]{
  height:auto!important;
  min-height:0!important;
}
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]>.stPlotlyChart{
  flex-shrink:0!important;
}
/* remove bottom margin on card when followed by plotly inside same column */
.card+div>[data-testid="stPlotlyChart"]{margin-top:0!important;}
            
.element-container {
    margin-bottom: 0rem !important;
}

div[data-testid="stVerticalBlock"] {
    gap: 0rem !important;
}

.stPlotlyChart {
    margin-top: -10px !important;
    margin-bottom: -10px !important;
}

.card {
    margin-bottom: 6px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    with open("heart_disease_prediction_model.pkl", "rb") as f:
        mdl = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        fnames = pickle.load(f)
    return mdl, fnames

model, feature_names = load_model()

FEATURE_LABELS = {
    "age":"Age","sex":"Sex","chest_pain_type":"Chest Pain",
    "resting_blood_pressure":"Resting BP","cholesterol":"Cholesterol",
    "fbs":"Blood Sugar","restecg":"Resting ECG",
    "max_heart_rate":"Max Heart Rate","exercise_induced_angina":"Exer. Angina",
    "oldpeak":"ST Depression","slope":"ST Slope",
    "major_vessels":"Major Vessels","thalassemia":"Thalassemia",
}

# ── Top Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#FFFFFF;border-bottom:1px solid #E2E8F0;
            box-shadow:0 2px 12px rgba(15,23,42,0.06);
            padding:15px 28px;display:flex;align-items:center;
            justify-content:space-between;flex-wrap:wrap;gap:12px;
            position:sticky;top:0;z-index:9999;">
  <div style="display:flex;align-items:center;gap:13px;">
    <div style="width:44px;height:44px;border-radius:13px;
                background:linear-gradient(135deg,#FEE2E2,#FECDD3);
                display:flex;align-items:center;justify-content:center;
                font-size:22px;box-shadow:0 4px 12px rgba(220,38,38,0.18);">❤️</div>
    <div>
      <div style="font-size:19px;font-weight:900;color:#0F172A;letter-spacing:-0.4px;line-height:1.1;">
        Heart Disease Risk Prediction
      </div>
      <div style="font-size:11px;color:#94A3B8;font-weight:500;margin-top:2px;">
        AI Powered &nbsp;·&nbsp; Explainable &nbsp;·&nbsp; Reliable
      </div>
    </div>
  </div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;">
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:11px;padding:9px 15px;min-width:130px;">
      <div style="font-size:9px;color:#94A3B8;font-weight:700;text-transform:uppercase;letter-spacing:.09em;margin-bottom:3px;">Model</div>
      <div style="font-size:13px;font-weight:700;color:#6D28D9;display:flex;align-items:center;gap:5px;">⚙️ XGBoost Classifier</div>
    </div>
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:11px;padding:9px 15px;min-width:165px;">
      <div style="font-size:9px;color:#94A3B8;font-weight:700;text-transform:uppercase;letter-spacing:.09em;margin-bottom:3px;">Model Status</div>
      <div style="font-size:13px;font-weight:700;color:#16A34A;display:flex;align-items:center;gap:6px;">
        <span style="width:8px;height:8px;border-radius:50%;background:#22C55E;
                     box-shadow:0 0 0 3px rgba(34,197,94,0.18);display:inline-block;"></span>
        Model Loaded Successfully
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Layout columns ──────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 3], gap="small")

# ── LEFT PANEL ──────────────────────────────────────────────────────────────────
with left_col:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#FFF1F2,#FFE4E6);
                padding:18px 18px 14px;border-bottom:1px solid #FECDD3;">
      <div style="display:flex;align-items:center;gap:9px;">
        <div style="width:32px;height:32px;border-radius:9px;background:rgba(220,38,38,0.12);
                    display:flex;align-items:center;justify-content:center;font-size:15px;">🧑‍⚕️</div>
        <div>
          <div style="font-size:14px;font-weight:800;color:#BE123C;line-height:1.2;">Patient Input</div>
          <div style="font-size:10px;color:#9F1239;margin-top:1px;">Enter patient details to predict risk</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    def sec(icon, lbl):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:7px;padding:13px 18px 5px;">
          <div style="width:20px;height:20px;border-radius:6px;background:#EFF6FF;
                      display:flex;align-items:center;justify-content:center;font-size:10px;">{icon}</div>
          <span style="font-size:10px;font-weight:700;color:#94A3B8;letter-spacing:.1em;text-transform:uppercase;">{lbl}</span>
        </div>""", unsafe_allow_html=True)

    def pw(): st.markdown('<div style="padding:0 18px;">', unsafe_allow_html=True)
    def pc(): st.markdown('</div>', unsafe_allow_html=True)
    def div(): st.markdown('<div style="height:1px;background:#F1F5F9;margin:2px 18px;"></div>', unsafe_allow_html=True)

    sec("👤", "Demographics")
    pw()
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
    sex = st.selectbox("Sex", ["Male","Female"])
    pc(); div()

    sec("💊", "Symptoms")
    cp_map   = {"Typical Angina":0,"Atypical Angina":1,"Non-Anginal Pain":2,"Asymptomatic":3}
    exang_map = {"No":0,"Yes":1}
    pw()
    cp    = st.selectbox("Chest Pain Type", list(cp_map.keys()))
    exang = st.selectbox("Exercise Induced Angina", list(exang_map.keys()))
    pc(); div()

    sec("❤️", "Vital Signs")
    pw()
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120, step=1)
    thalach  = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150, step=1)
    oldpeak  = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    pc(); div()

    sec("🧪", "Laboratory Results")
    fbs_map = {"No (≤120 mg/dl)":0,"Yes (>120 mg/dl)":1}
    pw()
    chol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=600, value=200, step=1)
    fbs  = st.selectbox("Fasting Blood Sugar > 120 mg/dl", list(fbs_map.keys()))
    pc(); div()

    sec("📊", "ECG & Imaging")
    restecg_map = {"Normal":0,"ST-T Wave Abnormality":1,"LV Hypertrophy":2}
    slope_map   = {"Downsloping":0,"Flat":1,"Upsloping":2}
    thal_map    = {"Normal":1,"Fixed Defect":2,"Reversible Defect":3}
    pw()
    restecg = st.selectbox("Resting ECG", list(restecg_map.keys()))
    slope   = st.selectbox("Slope of ST Segment", list(slope_map.keys()))
    ca      = st.selectbox("Number of Major Vessels (0–3)", [0,1,2,3])
    thal    = st.selectbox("Thalassemia", list(thal_map.keys()))
    pc()

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    pw()
    predict_clicked = st.button("🔍  Predict Risk", use_container_width=True)
    st.markdown("""<div style="font-size:10px;color:#94A3B8;text-align:center;margin-top:8px;line-height:1.5;">
      ⚕️ For educational purposes only.<br>Not a substitute for medical advice.</div>""", unsafe_allow_html=True)
    pc()
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)


# ── Chart helpers ───────────────────────────────────────────────────────────────
def gauge_chart(prob, title="📊 Risk Overview", sub="Probability of Heart Disease"):
    clr = "#16A34A" if prob < 35 else ("#F59E0B" if prob < 60 else ("#F97316" if prob < 80 else "#DC2626"))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob,
        number={"suffix":"%","font":{"size":32,"color":clr,"family":"Inter","weight":"bold"}},
        domain={"x":[0,1],"y":[0,0.85]},
        gauge={
            "axis":{"range":[0,100],"tickvals":[0,25,50,75,100],
                    "ticktext":["0%","25%","50%","75%","100%"],
                    "tickfont":{"size":9,"color":"#94A3B8"},"tickwidth":1,"tickcolor":"#CBD5E1"},
            "bar":{"color":clr,"thickness":0.24},
            "bgcolor":"#F1F5F9","borderwidth":0,
            "steps":[{"range":[0,35],"color":"#DCFCE7"},
                     {"range":[35,60],"color":"#FEF9C3"},
                     {"range":[60,80],"color":"#FFEDD5"},
                     {"range":[80,100],"color":"#FEE2E2"}],
            "threshold":{"line":{"color":"#0F172A","width":3},"thickness":0.85,"value":prob}
        }
    ))
    fig.add_annotation(x=0.02, y=1.06, xref="paper", yref="paper", xanchor="left", yanchor="top",
        text=f"<b>{title}</b>", font={"size":13,"color":"#1E293B","family":"Inter"}, showarrow=False)
    fig.add_annotation(x=0.02, y=0.97, xref="paper", yref="paper", xanchor="left", yanchor="top",
        text=sub, font={"size":11,"color":"#94A3B8","family":"Inter"}, showarrow=False)
    fig.add_annotation(x=0.02, y=0.02, xref="paper", yref="paper", xanchor="left", yanchor="bottom",
        text="0%", font={"size":9,"color":"#CBD5E1","family":"Inter"}, showarrow=False)
    fig.add_annotation(x=0.98, y=0.02, xref="paper", yref="paper", xanchor="right", yanchor="bottom",
        text="100%", font={"size":9,"color":"#CBD5E1","family":"Inter"}, showarrow=False)
    fig.update_layout(
        height=220,
        margin=dict(t=56,b=18,l=18,r=18),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family":"Inter"},
        shapes=[dict(type="rect",xref="paper",yref="paper",
                     x0=0,y0=0,x1=1,y1=1,
                     line=dict(color="#E2E8F0",width=1),
                     fillcolor="rgba(0,0,0,0)",
                     layer="below")]
    )
    return fig

def shap_bar(shap_vals, feat_names, top_n=10):
    sv = shap_vals[0] if len(shap_vals.shape)>1 else shap_vals
    pairs = sorted(zip(feat_names,sv), key=lambda x:abs(x[1]), reverse=True)[:top_n]
    labels = [FEATURE_LABELS.get(f,f) for f,_ in pairs]
    values = [v for _,v in pairs]
    colors = ["#EF4444" if v>0 else "#3B82F6" for v in values]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v:+.3f}" for v in values], textposition="outside",
        textfont={"size":9,"color":"#475569"},
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>"))
    fig.update_layout(height=240, margin=dict(t=4,b=30,l=4,r=54),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="SHAP Value (Impact)",title_font={"size":10,"color":"#94A3B8"},
                   tickfont={"size":9},zeroline=True,zerolinecolor="#E2E8F0",
                   zerolinewidth=2,gridcolor="#F8FAFC"),
        yaxis=dict(autorange="reversed",tickfont={"size":10,"color":"#374151"}),
        font={"family":"Inter"},showlegend=False)
    return fig, labels, values

def lime_bar(lime_list, top_n=10):
    pairs = sorted(lime_list, key=lambda x:abs(x[1]), reverse=True)[:top_n]
    raw, values = [p[0] for p in pairs], [p[1] for p in pairs]
    clean = []
    for lbl in raw:
        matched = lbl
        for k,v in FEATURE_LABELS.items():
            if k in lbl.lower(): matched = v; break
        clean.append(matched)
    colors = ["#10B981" if v>0 else "#EF4444" for v in values]
    fig = go.Figure(go.Bar(x=values, y=clean, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v:+.3f}" for v in values], textposition="outside",
        textfont={"size":9,"color":"#475569"},
        hovertemplate="<b>%{y}</b><br>LIME: %{x:.4f}<extra></extra>"))
    fig.update_layout(height=240, margin=dict(t=4,b=30,l=4,r=54),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="LIME Weight",title_font={"size":10,"color":"#94A3B8"},
                   tickfont={"size":9},zeroline=True,zerolinecolor="#E2E8F0",
                   zerolinewidth=2,gridcolor="#F8FAFC"),
        yaxis=dict(autorange="reversed",tickfont={"size":10,"color":"#374151"}),
        font={"family":"Inter"},showlegend=False)
    return fig, clean, values

def radar_chart(sl, sv, ll, lv, top_n=8):
    common = sl[:top_n]
    sn = [abs(v) for v in sv[:top_n]]; mx = max(sn+[1e-9]); sn = [v/mx for v in sn]
    ld = dict(zip(ll,[abs(v) for v in lv])); lmx = max(list(ld.values())+[1e-9])
    ln = [ld.get(l,0)/lmx for l in common]
    theta = common + [common[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=sn+[sn[0]],theta=theta,fill="toself",name="SHAP",
        line=dict(color="#3B82F6",width=2.5),fillcolor="rgba(59,130,246,0.12)"))
    fig.add_trace(go.Scatterpolar(r=ln+[ln[0]],theta=theta,fill="toself",name="LIME",
        line=dict(color="#10B981",width=2.5),fillcolor="rgba(16,185,129,0.12)"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True,range=[0,1],tickfont={"size":8},
                                   gridcolor="#E2E8F0",linecolor="#E2E8F0"),
                   angularaxis=dict(tickfont={"size":9,"color":"#475569"})),
        showlegend=True,
        legend=dict(orientation="h",y=-0.2,x=0.5,xanchor="center",
                    font={"size":10},bgcolor="rgba(0,0,0,0)"),
        height=230, margin=dict(t=8,b=38,l=28,r=28),
        paper_bgcolor="rgba(0,0,0,0)", font={"family":"Inter"})
    return fig


# ── RIGHT PANEL ─────────────────────────────────────────────────────────────────
with right_col:

    # ── Empty / welcome state ───────────────────────────────────────────────────
    if not predict_clicked:
        st.markdown("""
        <div style="min-height:40vh;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;gap:22px;text-align:center;padding:40px 20px;">
          <div style="width:96px;height:96px;border-radius:28px;
                      background:linear-gradient(135deg,#FEF2F2,#FECACA);
                      display:flex;align-items:center;justify-content:center;font-size:50px;
                      box-shadow:0 20px 48px rgba(239,68,68,0.14);">🫀</div>
          <div>
            <div style="font-size:28px;font-weight:900;color:#0F172A;letter-spacing:-0.5px;">
              Ready to Analyse
            </div>
            <div style="font-size:13px;color:#64748B;max-width:440px;line-height:1.8;margin:10px auto 0;">
              Fill in the patient details on the left panel and click
              <strong style="color:#DC2626;">Predict Risk</strong>
              to generate the full explainability report.
            </div>
          </div>
          <div style="display:flex;gap:14px;flex-wrap:wrap;justify-content:center;margin-top:6px;">
            <div style="background:#EFF6FF;border:1px solid #DBEAFE;border-radius:14px;padding:16px 20px;min-width:120px;">
              <div style="font-size:22px;">🤖</div>
              <div style="font-size:11px;font-weight:700;color:#2563EB;margin-top:6px;">XGBoost Model</div>
            </div>
            <div style="background:#F0FDF4;border:1px solid #DCFCE7;border-radius:14px;padding:16px 20px;min-width:120px;">
              <div style="font-size:22px;">🔬</div>
              <div style="font-size:11px;font-weight:700;color:#16A34A;margin-top:6px;">SHAP Analysis</div>
            </div>
            <div style="background:#FFFBEB;border:1px solid #FEF3C7;border-radius:14px;padding:16px 20px;min-width:120px;">
              <div style="font-size:22px;">🧪</div>
              <div style="font-size:11px;font-weight:700;color:#D97706;margin-top:6px;">LIME Analysis</div>
            </div>
            <div style="background:#FAF5FF;border:1px solid #EDE9FE;border-radius:14px;padding:16px 20px;min-width:120px;">
              <div style="font-size:22px;">📡</div>
              <div style="font-size:11px;font-weight:700;color:#7C3AED;margin-top:6px;">Radar Comparison</div>
            </div>
          </div>
          <div style="font-size:10.5px;color:#CBD5E1;margin-top:4px;">
            ⚕️ Educational &amp; research use only — not a substitute for medical advice
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Build input & run prediction ────────────────────────────────────────
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
            import lime, lime.lime_tabular
            bg  = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
            exp = lime.lime_tabular.LimeTabularExplainer(
                bg.values, feature_names=feature_names,
                class_names=["No Disease","Disease"],
                mode="classification", discretize_continuous=True)
            lr   = exp.explain_instance(input_df.values[0], model.predict_proba,
                                        num_features=10, top_labels=1)
            lkey = list(lr.local_exp.keys())[0]
            ll   = [(feature_names[i], w) for i, w in lr.local_exp[lkey]]
            lime_fig, lime_labels, lime_values = lime_bar(ll)
        except Exception:
            pass

        if shap_fig and lime_fig:
            try:
                radar_fig = radar_chart(shap_labels, shap_values_row, lime_labels, lime_values)
            except Exception:
                pass

        # ── ROW 1 ────────────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns([1.2, 1.0, 1.15], gap="small")

        with c1:
            # Single card: title in HTML, gauge fills it, no extra markdown after
            st.markdown("""
            <div class="card" id="gauge-card" style="padding-bottom:8px;">
              <div class="ctitle">📊 Risk Overview</div>
              <div class="csub">Probability of Heart Disease</div>
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(prob_d), use_container_width=True,
                            config={"displayModeBar": False})
            st.markdown(f"""<div style="text-align:center;margin-top:-18px;padding-bottom:8px;
                        font-size:9px;color:#CBD5E1;display:flex;justify-content:space-between;
                        padding-left:20px;padding-right:20px;">
              <span>0%</span><span>100%</span></div>""", unsafe_allow_html=True)

        with c2:
            pred_txt = ("The model predicts the patient is <b>likely to have heart disease</b>."
                        if prediction==1 else
                        "The model predicts the patient is <b>unlikely to have heart disease</b>.")
            st.markdown(f"""
            <div class="card" style="height:100%;">
              <div style="font-size:10px;font-weight:700;color:#94A3B8;text-transform:uppercase;
                          letter-spacing:.09em;margin-bottom:10px;">Risk Level</div>
              <div style="background:{r_bg};border:1.5px solid {r_brd};border-radius:12px;padding:14px;">
                <div style="font-size:20px;font-weight:900;color:{r_clr};margin-bottom:6px;">{r_icon} {r_lbl}</div>
                <div style="font-size:11px;color:#475569;line-height:1.65;">{pred_txt}</div>
              </div>
              <div style="margin-top:14px;display:flex;flex-direction:column;gap:9px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-size:11px;color:#64748B;font-weight:500;">Prediction</span>
                  <span style="font-size:12px;font-weight:700;color:{r_clr};">
                    {"1 (Disease)" if prediction==1 else "0 (No Disease)"}
                  </span>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-size:11px;color:#64748B;font-weight:500;">Confidence</span>
                  <span style="font-size:12px;font-weight:700;color:{conf_clr};">{conf_lbl}</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="card" style="height:100%;">
              <div style="font-size:13px;font-weight:700;color:#2563EB;margin-bottom:14px;
                          display:flex;align-items:center;gap:7px;">📋 Prediction Summary</div>
              <div class="srow">
                <span class="slbl" style="display:flex;align-items:center;gap:6px;">
                  <span style="color:#DC2626;font-size:13px;">●</span> Probability (Disease)
                </span>
                <span style="font-size:15px;font-weight:800;color:#DC2626;">{prob_d:.2f}%</span>
              </div>
              <div class="srow">
                <span class="slbl" style="display:flex;align-items:center;gap:6px;">
                  <span style="color:#16A34A;font-size:13px;">●</span> Probability (No Disease)
                </span>
                <span style="font-size:15px;font-weight:800;color:#16A34A;">{prob_nd:.2f}%</span>
              </div>
              <div class="srow">
                <span class="slbl">⚙️ Model Used</span>
                <span style="font-size:11px;font-weight:600;color:#374151;">XGBoost Classifier</span>
              </div>
              <div class="srow">
                <span class="slbl">📈 Model Accuracy</span>
                <span style="font-size:11px;font-weight:600;color:#374151;">~98.54%</span>
              </div>
              <div class="srow">
                <span class="slbl">⭐ Confidence</span>
                <span style="font-size:12px;font-weight:700;color:{conf_clr};">{conf_lbl}</span>
              </div>
            </div>""", unsafe_allow_html=True)

        # ── ROW 2: Explainability ─────────────────────────────────────────────────
        st.markdown("""
        <div class="secbar" style="margin-top:6px;">
          <div class="secbar-line" style="background:linear-gradient(135deg,#6366F1,#8B5CF6);"></div>
          <div class="secbar-txt">Explainability Dashboard</div>
        </div>""", unsafe_allow_html=True)

        e1, e2, e3 = st.columns(3, gap="small")

        with e1:
            st.markdown("""<div class="card">
              <div class="ctitle">🔷 SHAP Explanation</div>
              <div class="csub">Top Features (Impact on Prediction)</div>""", unsafe_allow_html=True)
            if shap_fig:
                st.plotly_chart(shap_fig, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown("""
                <div style="display:flex;gap:14px;justify-content:center;font-size:10px;
                            color:#64748B;padding-bottom:4px;">
                  <span style="display:flex;align-items:center;gap:4px;">
                    <i style="width:9px;height:9px;background:#EF4444;border-radius:2px;display:inline-block;"></i>
                    Increase Risk</span>
                  <span style="display:flex;align-items:center;gap:4px;">
                    <i style="width:9px;height:9px;background:#3B82F6;border-radius:2px;display:inline-block;"></i>
                    Decrease Risk</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Install SHAP: `pip install shap`")
            st.markdown("</div>", unsafe_allow_html=True)

        with e2:
            st.markdown("""<div class="card">
              <div class="ctitle">🟩 LIME Explanation</div>
              <div class="csub">Local Feature Importance (LIME)</div>""", unsafe_allow_html=True)
            if lime_fig:
                st.plotly_chart(lime_fig, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown("""
                <div style="display:flex;gap:14px;justify-content:center;font-size:10px;
                            color:#64748B;padding-bottom:4px;">
                  <span style="display:flex;align-items:center;gap:4px;">
                    <i style="width:9px;height:9px;background:#10B981;border-radius:2px;display:inline-block;"></i>
                    Positive Impact</span>
                  <span style="display:flex;align-items:center;gap:4px;">
                    <i style="width:9px;height:9px;background:#EF4444;border-radius:2px;display:inline-block;"></i>
                    Negative Impact</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Install LIME: `pip install lime`")
            st.markdown("</div>", unsafe_allow_html=True)

        with e3:
            st.markdown("""<div class="card">
              <div class="ctitle">📡 SHAP vs LIME Comparison</div>
              <div class="csub">Feature Importance Radar</div>""", unsafe_allow_html=True)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.info("Requires both SHAP and LIME installed.")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── ROW 3: Feature Impact + Recommendation ────────────────────────────────
        st.markdown("""
        <div class="secbar">
          <div class="secbar-line" style="background:linear-gradient(135deg,#F59E0B,#EF4444);"></div>
          <div class="secbar-txt">Feature Impact Summary &amp; Recommendation</div>
        </div>""", unsafe_allow_html=True)

        # Build factor lists from SHAP or fallback heuristics
        risk_factors, safe_factors = [], []
        if shap_labels and shap_values_row:
            for lbl, v in zip(shap_labels, shap_values_row):
                if v > 0.01:   risk_factors.append(lbl)
                elif v < -0.01: safe_factors.append(lbl)
        else:
            if vals["oldpeak"] > 1.5:                   risk_factors.append("High ST Depression")
            if vals["cholesterol"] > 240:               risk_factors.append("High Cholesterol")
            if vals["exercise_induced_angina"] == 1:    risk_factors.append("Exercise Induced Angina")
            if cp_map[cp] == 3:                         risk_factors.append("Asymptomatic Chest Pain")
            if vals["max_heart_rate"] > 140:            safe_factors.append("High Max Heart Rate")
            if vals["restecg"] == 0:                    safe_factors.append("Normal Resting ECG")
            if vals["major_vessels"] == 0:              safe_factors.append("No Major Vessel Blockage")
            if vals["exercise_induced_angina"] == 0:    safe_factors.append("No Exercise Induced Angina")

        r_pills = "".join([f'<span class="pill pr">▲ {f}</span>' for f in risk_factors[:6]]) or \
                  '<span class="pill pg">✓ No significant risk-increasing factors</span>'
        s_pills = "".join([f'<span class="pill pg">▼ {f}</span>' for f in safe_factors[:6]]) or \
                  '<span class="pill pr">⚠ No significant protective factors detected</span>'

        fi1, fi2 = st.columns([1, 1.1], gap="small")

        with fi1:
            st.markdown(f"""
            <div class="card">
              <div style="display:flex;gap:16px;">
                <div style="flex:1;padding-right:14px;border-right:1px solid #F1F5F9;">
                  <div style="font-size:11.5px;font-weight:700;color:#DC2626;margin-bottom:10px;
                              display:flex;align-items:center;gap:6px;">
                    <span style="background:#FEE2E2;border-radius:7px;width:22px;height:22px;
                                 display:inline-flex;align-items:center;justify-content:center;font-size:11px;">⬆</span>
                    Risk Increasing Factors
                  </div>
                  <div>{r_pills}</div>
                </div>
                <div style="flex:1;padding-left:2px;">
                  <div style="font-size:11.5px;font-weight:700;color:#16A34A;margin-bottom:10px;
                              display:flex;align-items:center;gap:6px;">
                    <span style="background:#DCFCE7;border-radius:7px;width:22px;height:22px;
                                 display:inline-flex;align-items:center;justify-content:center;font-size:11px;">⬇</span>
                    Risk Decreasing Factors
                  </div>
                  <div>{s_pills}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Recommendation content
        if prob_d >= 70:
            rc, ri, rt = "rec-h", "🚨", "Immediate Medical Attention Required"
            rx = ("This patient is at <strong>HIGH RISK</strong> of heart disease. "
                  "Immediate consultation with a cardiologist is strongly recommended. "
                  "Please schedule urgent diagnostic tests including ECG, echocardiogram, "
                  "and coronary angiography.")
        elif prob_d >= 40:
            rc, ri, rt = "rec-m", "⚠️", "Further Medical Evaluation Recommended"
            rx = ("This patient shows <strong>MODERATE RISK</strong> indicators. "
                  "Further evaluation is recommended — consider a lipid profile, "
                  "stress test, and lifestyle counseling.")
        else:
            rc, ri, rt = "rec-l", "✅", "Low Risk — Continue Preventive Care"
            rx = ("This patient shows <strong>LOW RISK</strong> indicators. "
                  "Continue healthy lifestyle: regular exercise, balanced diet, "
                  "and routine annual check-ups.")

        with fi2:
            st.markdown(f"""
            <div class="card" style="height:100%;">
              <div style="font-size:13px;font-weight:700;color:#1E293B;margin-bottom:12px;
                          display:flex;align-items:center;gap:7px;">🛡️ Recommendation</div>
              <div class="{rc}">
                <div style="font-size:13px;font-weight:700;color:#1E293B;margin-bottom:7px;">{ri} {rt}</div>
                <div style="font-size:12px;color:#374151;line-height:1.75;">{rx}</div>
              </div>
            </div>""", unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1E293B 0%,#0F172A 100%);
            border-radius:16px;padding:18px 28px;margin:6px 20px 24px;
            display:flex;justify-content:space-between;align-items:center;
            flex-wrap:wrap;gap:14px;">
  <div style="display:flex;gap:28px;flex-wrap:wrap;">
    <div>
      <div style="font-size:9px;color:#475569;font-weight:600;text-transform:uppercase;letter-spacing:.1em;">Dataset</div>
      <div style="font-size:11.5px;color:#E2E8F0;font-weight:600;margin-top:3px;">UCI Heart Disease Dataset</div>
    </div>
    <div>
      <div style="font-size:9px;color:#475569;font-weight:600;text-transform:uppercase;letter-spacing:.1em;">Model</div>
      <div style="font-size:11.5px;color:#E2E8F0;font-weight:600;margin-top:3px;">XGBoost Classifier</div>
    </div>
    <div>
      <div style="font-size:9px;color:#475569;font-weight:600;text-transform:uppercase;letter-spacing:.1em;">Explainability</div>
      <div style="font-size:11.5px;color:#E2E8F0;font-weight:600;margin-top:3px;">SHAP + LIME</div>
    </div>
    <div>
      <div style="font-size:9px;color:#475569;font-weight:600;text-transform:uppercase;letter-spacing:.1em;">Project</div>
      <div style="font-size:11.5px;color:#E2E8F0;font-weight:600;margin-top:3px;">Explainable AI for Healthcare Risk Prediction</div>
    </div>
  </div>
  # <div style="text-align:right;">
  #   <div style="font-size:10px;color:#475569;margin-bottom:3px;">Developer</div>
  #   <div style="font-size:12px;font-weight:700;color:#60A5FA;">Final Year AI/ML Project</div>
  #   <div style="font-size:10px;color:#475569;margin-top:4px;">⚕️ Educational Use Only — Not a Medical Device</div>
  # </div>
</div>
""", unsafe_allow_html=True)
