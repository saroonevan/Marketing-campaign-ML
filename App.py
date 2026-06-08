import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title = "CampaignAI",
    page_icon  = "🚀",
    layout     = "wide",
    initial_sidebar_state = "collapsed"
)

# ══════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Outfit:wght@300;400;500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 10% 5%,  rgba(236,98,55,0.09) 0%, transparent 45%),
        radial-gradient(ellipse at 90% 95%, rgba(46,196,182,0.07) 0%, transparent 45%),
        #0d0b13;
}
html, body, [class*="css"] { font-family:'Outfit',sans-serif; }
#MainMenu, footer, header   { visibility:hidden; }
.block-container             { padding-top:1.8rem; max-width:1200px; }

/* ── Hero ── */
.hero        { text-align:center; padding:0.5rem 0 0.2rem; }
.hero-badge  {
    display:inline-block; font-size:0.68rem; letter-spacing:0.22em;
    text-transform:uppercase; color:#ec6237;
    border:1px solid rgba(236,98,55,0.3); border-radius:100px;
    padding:0.3rem 1rem; margin-bottom:0.9rem; font-weight:500;
}
.hero-title  {
    font-family:'Fraunces',serif; font-size:3rem; font-weight:600;
    line-height:1.05; color:#f5f1ea; margin:0; letter-spacing:-0.02em;
}
.hero-title .acc { font-style:italic; color:#ec6237; }
.hero-sub    { font-size:1rem; color:#9b94a8; margin-top:0.6rem; font-weight:300; }

/* ── KPI strip ── */
div[data-testid="stMetric"] {
    background: linear-gradient(160deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02));
    border:1px solid rgba(155,148,168,0.15);
    border-radius:14px; padding:1.1rem 1.4rem;
    transition: border-color 0.2s;
}
div[data-testid="stMetric"]:hover { border-color:rgba(236,98,55,0.45); }
div[data-testid="stMetricLabel"]  { color:#9b94a8 !important; font-size:0.78rem !important; letter-spacing:0.1em; }
div[data-testid="stMetricValue"]  {
    font-family:'Fraunces',serif !important;
    color:#f5f1ea !important; font-size:1.9rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap:0.4rem; background:rgba(255,255,255,0.025);
    padding:0.35rem; border-radius:14px;
    border:1px solid rgba(155,148,168,0.12);
}
.stTabs [data-baseweb="tab"] {
    height:44px; padding:0 1.6rem;
    background:transparent; border-radius:10px;
    color:#9b94a8; font-weight:500; font-size:0.92rem; border:none;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#ec6237,#c94a24) !important;
    color:#fff !important;
    box-shadow:0 4px 18px rgba(236,98,55,0.3);
}

/* ── Inputs ── */
.stSelectbox label,.stSlider label,.stNumberInput label,.stMultiSelect label {
    color:#c8c2d4 !important; font-size:0.82rem !important; font-weight:400 !important;
}
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput div[data-baseweb="input"],
.stMultiSelect div[data-baseweb="select"] > div {
    background:rgba(255,255,255,0.035) !important;
    border:1px solid rgba(155,148,168,0.18) !important;
    border-radius:10px !important;
}

/* ── Group label ── */
.grp {
    font-size:0.68rem; letter-spacing:0.2em; text-transform:uppercase;
    color:#ec6237; font-weight:600; margin-bottom:0.5rem;
    padding-bottom:0.4rem; border-bottom:1px solid rgba(155,148,168,0.12);
}

/* ── Button ── */
.stButton > button {
    background:linear-gradient(135deg,#ec6237,#c94a24) !important;
    color:#fff !important; border:none !important;
    border-radius:12px !important; font-weight:600 !important;
    font-size:1rem !important; letter-spacing:0.02em !important;
    box-shadow:0 6px 22px rgba(236,98,55,0.3);
    transition:all 0.2s ease !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 10px 30px rgba(236,98,55,0.45) !important;
}

/* ── Result card ── */
.rcard {
    background:linear-gradient(160deg,rgba(255,255,255,0.05),rgba(255,255,255,0.02));
    border:1px solid rgba(155,148,168,0.15);
    border-radius:18px; padding:1.6rem; margin-top:0.5rem;
}
.rcard-profit { border-color:rgba(46,196,182,0.4); }
.rcard-loss   { border-color:rgba(231,76,60,0.4);  }
.big-val { font-family:'Fraunces',serif; font-size:2.6rem; font-weight:600; line-height:1; margin:0.25rem 0; }
.col-green  { color:#2ec4b6; }
.col-orange { color:#ec6237; }
.col-red    { color:#e74c3c; }
.mlabel { font-size:0.68rem; letter-spacing:0.15em; text-transform:uppercase; color:#6b6577; }

/* ── Insight box ── */
.insight {
    background:rgba(255,255,255,0.03);
    border-left:3px solid #ec6237;
    border-radius:0 12px 12px 0;
    padding:1rem 1.2rem; margin-top:1rem;
    color:#c8c2d4; font-size:0.9rem; line-height:1.6;
}

/* ── Section title ── */
.stab-title {
    font-family:'Fraunces',serif; font-size:1.55rem;
    font-weight:600; color:#f5f1ea; margin:0.2rem 0 0.1rem; letter-spacing:-0.01em;
}
.stab-sub { color:#9b94a8; font-size:0.88rem; font-weight:300; margin-bottom:1rem; }

/* ── Expander ── */
.streamlit-expanderHeader { background:rgba(255,255,255,0.03) !important; border-radius:10px !important; }
hr { border-color:rgba(155,148,168,0.1); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

@st.cache_resource
def load_models():
    with open(os.path.join(MODEL_DIR,'regression_model.pkl'),'rb') as f:
        reg = pickle.load(f)
    with open(os.path.join(MODEL_DIR,'classification_model.pkl'),'rb') as f:
        clf = pickle.load(f)
    return reg, clf

try:
    reg_pkg, clf_pkg = load_models()
    models_loaded    = True
except Exception as e:
    models_loaded = False
    load_error    = str(e)

# ══════════════════════════════════════════════════════
# HERO + KPI
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <span class="hero-badge">✦ Nykaa · Purplle · Tira</span>
    <h1 class="hero-title">AI-powered campaign <span class="acc">intelligence</span></h1>
    <p class="hero-sub">Forecast revenue and predict profitability before your campaign goes live</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
k1,k2,k3,k4 = st.columns(4)
k1.metric("Total Campaigns", "166,665")
k2.metric("Revenue R² Score", "99.20%")
k3.metric("Profit Accuracy",  "97.28%")
k4.metric("Loss Recall",       "97.11%")
st.markdown("<br>", unsafe_allow_html=True)

if not models_loaded:
    st.error(f"Could not load models: {load_error}")
    st.stop()

# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════
def build_features(inputs, feature_cols, model_type='reg'):
    row = {col:0 for col in feature_cols}
    for k,v in [('Duration',inputs['duration']),('Impressions',inputs['impressions']),
                ('Clicks',inputs['clicks']),('Leads',inputs['leads']),
                ('Conversions',inputs['conversions']),('Acquisition_Cost',inputs['acq_cost']),
                ('Engagement_Score',inputs['engagement'])]:
        if k in row: row[k]=v
    if 'Updated_ROI' in row:
        est = inputs['conversions']*450
        row['Updated_ROI'] = (est-inputs['acq_cost'])/max(inputs['acq_cost'],1)
    ch_map = {'Email':'Email','Facebook':'Facebook','Google':'Google',
              'Instagram':'Instagram','WhatsApp':'WhatsApp','YouTube':'YouTube'}
    for ch in inputs['channels']:
        if ch in row: row[ch]=1
    bk = f"Brand_{inputs['brand']}"
    if bk in row: row[bk]=1
    ct = {'Social Media':'Campaign_Type_Social Media','Email':'Campaign_Type_Email',
          'SEO':'Campaign_Type_SEO','Influencer':'Campaign_Type_Influencer',
          'Paid Ads':'Campaign_Type_Paid Ads'}.get(inputs['campaign_type'],'')
    if ct in row: row[ct]=1
    ta = {'College Students':'Target_Audience_College Students',
          'Youth':'Target_Audience_Youth','Working Women':'Target_Audience_Working Women',
          'Premium Shoppers':'Target_Audience_Premium Shoppers',
          'Tier 2 City Customers':'Target_Audience_Tier 2 City Customers'}.get(inputs['target_audience'],'')
    if ta in row: row[ta]=1
    lk = {'Hindi':'Language_Hindi','English':'Language_English','Tamil':'Language_Tamil',
          'Bengali':'Language_Bengali','Native':'Language_Native'}.get(inputs['language'],'')
    if lk in row: row[lk]=1
    sk = {'College Students':'Customer_Segment_College Students',
          'Youth':'Customer_Segment_Youth','Working Women':'Customer_Segment_Working Women',
          'Premium Shoppers':'Customer_Segment_Premium Shoppers',
          'Tier 2 City Customers':'Customer_Segment_Tier 2 City Customers',
          'Basic Plan':'Customer_Segment_Basic Plan'}.get(inputs['customer_segment'],'')
    if sk in row: row[sk]=1
    return pd.DataFrame([row])[feature_cols]

def input_form(prefix):
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="grp">Brand & Campaign</div>', unsafe_allow_html=True)
        brand = st.selectbox("Brand",["Nykaa","Purplle","Tira"],key=f"{prefix}_b")
        campaign_type = st.selectbox("Campaign Type",
            ["Social Media","Email","SEO","Influencer","Paid Ads"],key=f"{prefix}_ct")
        target_audience = st.selectbox("Target Audience",
            ["College Students","Youth","Working Women","Premium Shoppers","Tier 2 City Customers"],
            key=f"{prefix}_ta")
        language = st.selectbox("Language",
            ["English","Hindi","Tamil","Bengali","Native"],key=f"{prefix}_lang")
        customer_segment = st.selectbox("Customer Segment",
            ["College Students","Youth","Working Women","Premium Shoppers","Tier 2 City Customers","Basic Plan"],
            key=f"{prefix}_seg")

        st.markdown('<div class="grp" style="margin-top:1rem;">Performance Metrics</div>', unsafe_allow_html=True)
        a,b = st.columns(2)
        impressions = a.number_input("Impressions",10000,100000,55000,step=1000,key=f"{prefix}_imp")
        clicks      = b.number_input("Clicks",200,15000,3900,step=100,key=f"{prefix}_clk")
        c,d         = st.columns(2)
        leads       = c.number_input("Leads",50,9000,1476,step=50,key=f"{prefix}_lds")
        conversions = d.number_input("Conversions",10,7000,775,step=50,key=f"{prefix}_conv")

    with col2:
        st.markdown('<div class="grp">Campaign Settings</div>', unsafe_allow_html=True)
        acq_cost   = st.number_input("Acquisition Cost (₹)",8.0,15000.0,376.0,step=10.0,key=f"{prefix}_cost")
        duration   = st.slider("Duration (days)",5,30,15,key=f"{prefix}_dur")
        engagement = st.slider("Engagement Score",0.0,30.0,13.5,step=0.1,key=f"{prefix}_eng")

        st.markdown('<div class="grp" style="margin-top:1rem;">Marketing Channels</div>', unsafe_allow_html=True)
        channels = st.multiselect(
            "Select channels used",
            ["Email","Facebook","Google","Instagram","WhatsApp","YouTube"],
            key=f"{prefix}_ch"
        )
        if channels:
            st.markdown("**Active:** " + "  ".join([f"`{ch}`" for ch in channels]))
        else:
            st.caption("No channels selected yet")

    return dict(brand=brand, campaign_type=campaign_type, target_audience=target_audience,
                language=language, customer_segment=customer_segment, duration=duration,
                impressions=impressions, clicks=clicks, leads=leads, conversions=conversions,
                acq_cost=acq_cost, engagement=engagement, channels=channels)

def validate(inputs):
    if not inputs['channels']:
        st.error("⚠️ Please select at least one marketing channel"); return False
    if inputs['conversions'] > inputs['leads']:
        st.error("⚠️ Conversions cannot exceed Leads"); return False
    if inputs['leads'] > inputs['clicks']:
        st.error("⚠️ Leads cannot exceed Clicks"); return False
    if inputs['clicks'] > inputs['impressions']:
        st.error("⚠️ Clicks cannot exceed Impressions"); return False
    if inputs['acq_cost'] <= 0:
        st.error("⚠️ Acquisition Cost must be greater than 0"); return False
    return True

def summary_table(inputs):
    df = pd.DataFrame({
        'Feature': ['Brand','Campaign Type','Target Audience','Language','Customer Segment',
                    'Duration','Impressions','Clicks','Leads','Conversions',
                    'Acquisition Cost','Engagement Score','Channels'],
        'Value'  : [inputs['brand'],inputs['campaign_type'],inputs['target_audience'],
                    inputs['language'],inputs['customer_segment'],
                    f"{inputs['duration']} days",f"{inputs['impressions']:,}",
                    f"{inputs['clicks']:,}",f"{inputs['leads']:,}",f"{inputs['conversions']:,}",
                    f"₹{inputs['acq_cost']:,.2f}",inputs['engagement'],
                    ', '.join(inputs['channels']) or 'None']
    })
    st.table(df)

# ══════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "💰  Revenue Prediction",
    "📈  Profit / Loss Prediction",
    "ℹ️  Model Insights"
])

# ════════════════════════════════════════
# TAB 1 — REVENUE
# ════════════════════════════════════════
with tab1:
    st.markdown('<div class="stab-title">Revenue Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="stab-sub">Estimate how much revenue this campaign will generate using XGBoost (R² 99.20%)</div>', unsafe_allow_html=True)

    inputs = input_form("reg")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀  Forecast Revenue", use_container_width=True, key="reg_btn"):
        if validate(inputs):
            with st.spinner("Running XGBoost model..."):
                X       = build_features(inputs, reg_pkg['features'], 'reg')
                revenue = max(0.0, float(reg_pkg['model'].predict(X)[0]))

            st.markdown("---")
            st.markdown("### 🎯 Revenue Prediction Result")
            r1, r2 = st.columns(2, gap="large")

            with r1:
                roi = (revenue-inputs['acq_cost'])/max(inputs['acq_cost'],1)
                if   revenue < 100000:  tier,col = "Low revenue · under ₹1 Lakh",   "col-red"
                elif revenue < 500000:  tier,col = "Average revenue · ₹1L – ₹5L",   "col-orange"
                elif revenue < 1000000: tier,col = "Good revenue · ₹5L – ₹10L",     "col-green"
                else:                   tier,col = "High revenue · above ₹10 Lakh",  "col-green"

                st.markdown(f"""
                <div class="rcard">
                    <div class="mlabel">Predicted Revenue</div>
                    <div class="big-val {col}">₹{revenue:,.0f}</div>
                    <p style="color:#9b94a8;margin:0.4rem 0 1.1rem;font-size:0.88rem;">{tier}</p>
                    <div style="display:flex;gap:2.5rem;flex-wrap:wrap;">
                        <div>
                            <div class="mlabel">Model R²</div>
                            <div style="color:#f5f1ea;font-size:1.25rem;font-weight:600;margin-top:2px;">
                                {reg_pkg['r2_score']*100:.2f}%
                            </div>
                        </div>
                        <div>
                            <div class="mlabel">Estimated ROI</div>
                            <div style="color:#f5f1ea;font-size:1.25rem;font-weight:600;margin-top:2px;">
                                {roi:.2f}×
                            </div>
                        </div>
                        <div>
                            <div class="mlabel">Avg Model Error</div>
                            <div style="color:#f5f1ea;font-size:1.25rem;font-weight:600;margin-top:2px;">
                                ₹{reg_pkg['rmse']:,.0f}
                            </div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="insight">
                    {'✅ <strong>Profitable campaign.</strong>' if roi > 0 else '❌ <strong>Loss-making campaign.</strong>'}
                    {'Proceed with confidence — estimated ROI is strong.' if roi > 2 else
                     'Campaign is profitable but consider optimising channels.' if roi > 0 else
                     'Review acquisition cost and channel strategy.'}
                </div>""", unsafe_allow_html=True)

            with r2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=revenue,
                    number={'prefix':"₹",'valueformat':",.0f",'font':{'color':'#f5f1ea','size':28}},
                    gauge={'axis':{'range':[0,2500000],'tickcolor':'#6b6577','tickfont':{'color':'#6b6577','size':9}},
                           'bar':{'color':'#ec6237'},'bgcolor':'rgba(0,0,0,0)','borderwidth':0,
                           'steps':[{'range':[0,100000],      'color':'rgba(231,76,60,0.2)'},
                                    {'range':[100000,500000],  'color':'rgba(243,156,18,0.2)'},
                                    {'range':[500000,1000000], 'color':'rgba(46,196,182,0.2)'},
                                    {'range':[1000000,2500000],'color':'rgba(46,196,182,0.3)'}]}))
                fig.update_layout(height=300, margin=dict(t=30,b=10,l=30,r=30),
                    paper_bgcolor='rgba(0,0,0,0)', font_color='#f5f1ea')
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋  Campaign Input Summary"):
                summary_table(inputs)

# ════════════════════════════════════════
# TAB 2 — PROFIT / LOSS
# ════════════════════════════════════════
with tab2:
    st.markdown('<div class="stab-title">Profit / Loss Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="stab-sub">Predict whether this campaign will be profitable using Logistic Regression + SMOTE (97.28% accuracy)</div>', unsafe_allow_html=True)

    inputs_clf = input_form("clf")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("📊  Analyse Profitability", use_container_width=True, key="clf_btn"):
        if validate(inputs_clf):
            with st.spinner("Running Logistic Regression model..."):
                X    = build_features(inputs_clf, clf_pkg['features'], 'clf')
                X_sc = clf_pkg['scaler'].transform(X)
                pred = clf_pkg['model'].predict(X_sc)[0]
                prob = clf_pkg['model'].predict_proba(X_sc)[0]
                classes  = list(clf_pkg['model'].classes_)
                p_profit = round(float(prob[classes.index(1)])*100,1)
                p_loss   = round(float(prob[classes.index(0)])*100,1)

            st.markdown("---")
            st.markdown("### 🎯 Profit / Loss Result")
            r1, r2 = st.columns(2, gap="large")

            with r1:
                is_profit = (int(pred)==1)
                card_cls  = "rcard rcard-profit" if is_profit else "rcard rcard-loss"
                val_cls   = "col-green" if is_profit else "col-red"
                verdict   = "Profitable ✓" if is_profit else "Loss ✗"
                conf      = p_profit if is_profit else p_loss
                conf_lbl  = f"Confidence: {conf}%"
                desc      = "This campaign is expected to generate profit." if is_profit \
                            else "This campaign is expected to result in a loss."

                st.markdown(f"""
                <div class="{card_cls}">
                    <div class="mlabel">Prediction</div>
                    <div class="big-val {val_cls}">{verdict}</div>
                    <p style="color:#9b94a8;margin:0.4rem 0 1.1rem;font-size:0.88rem;">{desc}</p>
                    <div style="display:flex;gap:2.5rem;flex-wrap:wrap;">
                        <div>
                            <div class="mlabel">Confidence</div>
                            <div style="color:{'#2ec4b6' if is_profit else '#e74c3c'};font-size:1.4rem;font-weight:600;margin-top:2px;">
                                {conf}%
                            </div>
                        </div>
                        <div>
                            <div class="mlabel">Model Accuracy</div>
                            <div style="color:#f5f1ea;font-size:1.4rem;font-weight:600;margin-top:2px;">
                                {clf_pkg['accuracy']*100:.1f}%
                            </div>
                        </div>
                        <div>
                            <div class="mlabel">Loss Recall</div>
                            <div style="color:#f5f1ea;font-size:1.4rem;font-weight:600;margin-top:2px;">
                                {clf_pkg['loss_recall']*100:.1f}%
                            </div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="insight">
                    {'✅ <strong>Go ahead!</strong> Campaign shows strong profit signals. Consider scaling budget on top channels.' 
                     if is_profit else
                     '⚠️ <strong>Review required.</strong> Consider switching to Influencer campaigns, targeting Premium Shoppers, or reducing acquisition cost.'}
                </div>""", unsafe_allow_html=True)

            with r2:
                fig = go.Figure(go.Pie(
                    labels=['Profit','Loss'], values=[p_profit,p_loss], hole=0.62,
                    marker_colors=['#2ec4b6','#e74c3c'],
                    textinfo='label+percent',
                    textfont={'color':'#f5f1ea','size':13},
                    hovertemplate="%{label}: %{value}%<extra></extra>"))
                fig.update_layout(
                    height=300, margin=dict(t=20,b=20,l=20,r=20),
                    paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
                    annotations=[{'text': "Profit" if is_profit else "Loss",
                                  'x':0.5,'y':0.5,'font_size':22,
                                  'font_color':'#2ec4b6' if is_profit else '#e74c3c',
                                  'showarrow':False}])
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋  Campaign Input Summary"):
                summary_table(inputs_clf)

# ════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ════════════════════════════════════════
with tab3:
    st.markdown('<div class="stab-title">Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="stab-sub">Performance metrics, training details and dataset overview</div>', unsafe_allow_html=True)

    m1, m2 = st.columns(2, gap="large")

    with m1:
        st.markdown("#### 💰 Regression Model")
        st.markdown(f"""
        <div class="rcard">
            <div class="mlabel">Model</div>
            <div style="color:#f5f1ea;font-size:1.1rem;font-weight:600;margin-bottom:1rem;">
                {reg_pkg['model_name']}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
                <div><div class="mlabel">R² Score</div>
                     <div style="color:#2ec4b6;font-size:1.4rem;font-weight:600;">{reg_pkg['r2_score']*100:.2f}%</div></div>
                <div><div class="mlabel">RMSE</div>
                     <div style="color:#f5f1ea;font-size:1.4rem;font-weight:600;">₹{reg_pkg['rmse']:,.0f}</div></div>
                <div><div class="mlabel">MAE</div>
                     <div style="color:#f5f1ea;font-size:1.4rem;font-weight:600;">₹{reg_pkg['mae']:,.0f}</div></div>
                <div><div class="mlabel">Train/Test Gap</div>
                     <div style="color:#f5f1ea;font-size:1.4rem;font-weight:600;">{reg_pkg['gap']:.4f}</div></div>
                <div><div class="mlabel">Trained On</div>
                     <div style="color:#f5f1ea;font-size:0.95rem;font-weight:500;">{reg_pkg['trained_on']}</div></div>
                <div><div class="mlabel">Features</div>
                     <div style="color:#f5f1ea;font-size:0.95rem;font-weight:500;">{len(reg_pkg['features'])} columns</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown("#### 📈 Classification Model")
        st.markdown(f"""
        <div class="rcard">
            <div class="mlabel">Model</div>
            <div style="color:#f5f1ea;font-size:1.1rem;font-weight:600;margin-bottom:1rem;">
                {clf_pkg['model_name']} + SMOTE
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
                <div><div class="mlabel">Accuracy</div>
                     <div style="color:#2ec4b6;font-size:1.4rem;font-weight:600;">{clf_pkg['accuracy']*100:.2f}%</div></div>
                <div><div class="mlabel">F1 Score</div>
                     <div style="color:#f5f1ea;font-size:1.4rem;font-weight:600;">{clf_pkg['f1_score']*100:.2f}%</div></div>
                <div><div class="mlabel">ROC-AUC</div>
                     <div style="color:#f5f1ea;font-size:1.4rem;font-weight:600;">{clf_pkg['roc_auc']:.4f}</div></div>
                <div><div class="mlabel">Loss Recall</div>
                     <div style="color:#ec6237;font-size:1.4rem;font-weight:600;">{clf_pkg['loss_recall']*100:.2f}%</div></div>
                <div><div class="mlabel">Trained On</div>
                     <div style="color:#f5f1ea;font-size:0.95rem;font-weight:500;">{clf_pkg['trained_on']}</div></div>
                <div><div class="mlabel">SMOTE Used</div>
                     <div style="color:#f5f1ea;font-size:0.95rem;font-weight:500;">{'Yes ✅' if clf_pkg['smote_used'] else 'No'}</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 Dataset Overview")
    d1,d2,d3,d4,d5 = st.columns(5)
    d1.metric("Total Campaigns",  "166,665")
    d2.metric("Brands",           "3")
    d3.metric("Features",         "38")
    d4.metric("Profit Campaigns", "95.01%")
    d5.metric("Loss Campaigns",   "4.99%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🏆 All Models Compared")

    reg_df = pd.DataFrame({
        'Model'    : ['Linear Regression','Ridge Regression','Lasso Regression',
                      'Decision Tree','Random Forest','XGBoost ✅'],
        'R² Score' : ['77.11%','77.11%','77.11%','99.21%','99.65%','99.20%'],
        'RMSE'     : ['₹2,22,882','₹2,22,882','₹2,22,882','₹41,483','₹27,550','₹41,744'],
        'Selected' : ['','','','','','✅ Winner']
    })
    st.markdown("**Regression Models**")
    st.dataframe(reg_df, use_container_width=True, hide_index=True)

    clf_df = pd.DataFrame({
        'Model'       : ['Logistic Regression ✅','Decision Tree','Random Forest',
                         'XGBoost','KNN','Naive Bayes','Gradient Boosting',
                         'AdaBoost','SVM','LightGBM'],
        'Accuracy'    : ['97.28%','100%*','100%*','99.90%','94.86%',
                         '99.99%','100%*','100%*','99.29%','100%*'],
        'Loss Recall' : ['97.11%','100%*','100%*','98.50%','0.00%',
                         '100%*','100%*','100%*','87.31%','100%*'],
        'Note'        : ['✅ Mentor approved','⚠️ Data leakage','⚠️ Data leakage',
                         'Good','Poor recall','⚠️ Leakage','⚠️ Leakage',
                         '⚠️ Leakage','Good','⚠️ Leakage']
    })
    st.markdown("**Classification Models** *(* = Updated_ROI leakage)*")
    st.dataframe(clf_df, use_container_width=True, hide_index=True)