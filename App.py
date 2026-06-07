import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title            = "Marketing Campaign Predictor",
    page_icon             = "📊",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

# ── Load models ───────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

@st.cache_resource
def load_models():
    with open(os.path.join(MODEL_DIR, 'regression_model.pkl'), 'rb') as f:
        reg_pkg = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'classification_model.pkl'), 'rb') as f:
        clf_pkg = pickle.load(f)
    return reg_pkg, clf_pkg

try:
    reg_pkg, clf_pkg = load_models()
    models_loaded    = True
except Exception as e:
    models_loaded = False
    load_error    = str(e)

# ── Sidebar info ──────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 CampaignAI")
    st.markdown("*Beauty Brand Analytics*")
    st.markdown("---")
    if models_loaded:
        st.success("Models loaded ✅")
        st.markdown("**Regression**")
        st.caption(f"Model  : {reg_pkg['model_name']}")
        st.caption(f"R²     : {reg_pkg['r2_score']*100:.2f}%")
        st.caption(f"RMSE   : ₹{reg_pkg['rmse']:,.0f}")
        st.markdown("**Classification**")
        st.caption(f"Model  : {clf_pkg['model_name']}")
        st.caption(f"Acc    : {clf_pkg['accuracy']*100:.2f}%")
        st.caption(f"Recall : {clf_pkg['loss_recall']*100:.2f}%")
    else:
        st.error("Models not loaded ❌")
    st.markdown("---")
    st.markdown("**3 Brands**")
    st.caption("Nykaa · Purplle · Tira")
    st.markdown("**166,665 Campaigns**")
    st.caption("Jul 2024 → Jun 2025")
    st.markdown("**38 Features**")
    st.caption("Brand, Channel, Audience...")

# ── Build feature vector ──────────────────────────────
def build_features(inputs, feature_cols, model_type='reg'):
    row = {col: 0 for col in feature_cols}

    if 'Duration'         in row: row['Duration']         = inputs['duration']
    if 'Impressions'      in row: row['Impressions']      = inputs['impressions']
    if 'Clicks'           in row: row['Clicks']           = inputs['clicks']
    if 'Leads'            in row: row['Leads']            = inputs['leads']
    if 'Conversions'      in row: row['Conversions']      = inputs['conversions']
    if 'Acquisition_Cost' in row: row['Acquisition_Cost'] = inputs['acq_cost']
    if 'Engagement_Score' in row: row['Engagement_Score'] = inputs['engagement']

    if 'Updated_ROI' in row:
        est = inputs['conversions'] * 450
        row['Updated_ROI'] = (
            est - inputs['acq_cost']
        ) / max(inputs['acq_cost'], 1)

    if 'Email'     in row: row['Email']     = inputs['email']
    if 'Facebook'  in row: row['Facebook']  = inputs['facebook']
    if 'Google'    in row: row['Google']    = inputs['google']
    if 'Instagram' in row: row['Instagram'] = inputs['instagram']
    if 'WhatsApp'  in row: row['WhatsApp']  = inputs['whatsapp']
    if 'YouTube'   in row: row['YouTube']   = inputs['youtube']

    bk = f"Brand_{inputs['brand']}"
    if bk in row: row[bk] = 1

    ct = {'Social Media':'Campaign_Type_Social Media',
          'Email':'Campaign_Type_Email','SEO':'Campaign_Type_SEO',
          'Influencer':'Campaign_Type_Influencer',
          'Paid Ads':'Campaign_Type_Paid Ads'}.get(inputs['campaign_type'],'')
    if ct in row: row[ct] = 1

    ta = {'College Students':'Target_Audience_College Students',
          'Youth':'Target_Audience_Youth',
          'Working Women':'Target_Audience_Working Women',
          'Premium Shoppers':'Target_Audience_Premium Shoppers',
          'Tier 2 City Customers':'Target_Audience_Tier 2 City Customers'
          }.get(inputs['target_audience'],'')
    if ta in row: row[ta] = 1

    lk = {'Hindi':'Language_Hindi','English':'Language_English',
          'Tamil':'Language_Tamil','Bengali':'Language_Bengali',
          'Native':'Language_Native'}.get(inputs['language'],'')
    if lk in row: row[lk] = 1

    sk = {'College Students':'Customer_Segment_College Students',
          'Youth':'Customer_Segment_Youth',
          'Working Women':'Customer_Segment_Working Women',
          'Premium Shoppers':'Customer_Segment_Premium Shoppers',
          'Tier 2 City Customers':'Customer_Segment_Tier 2 City Customers',
          'Basic Plan':'Customer_Segment_Basic Plan'
          }.get(inputs['customer_segment'],'')
    if sk in row: row[sk] = 1

    return pd.DataFrame([row])[feature_cols]

# ══════════════════════════════════════════════════════
# MAIN PAGE — ALL IN ONE
# ══════════════════════════════════════════════════════
st.title("📊 Marketing Campaign Predictor")
st.markdown("Fill in your campaign details once — get **Revenue** and **Profit/Loss** predictions together")
st.markdown("---")

if not models_loaded:
    st.error(f"Could not load models: {load_error}")
    st.stop()

# ── INPUT FORM ────────────────────────────────────────
st.markdown("### 📋 Campaign Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Brand & Campaign**")
    brand = st.selectbox("Brand",
        ["Nykaa","Purplle","Tira"])
    campaign_type = st.selectbox("Campaign Type",
        ["Social Media","Email","SEO","Influencer","Paid Ads"])
    target_audience = st.selectbox("Target Audience",
        ["College Students","Youth","Working Women",
         "Premium Shoppers","Tier 2 City Customers"])
    language = st.selectbox("Language",
        ["Hindi","English","Tamil","Bengali","Native"])
    customer_segment = st.selectbox("Customer Segment",
        ["College Students","Youth","Working Women",
         "Premium Shoppers","Tier 2 City Customers","Basic Plan"])

with col2:
    st.markdown("**Performance Numbers**")
    duration    = st.slider("Duration (days)", 5, 30, 15)
    impressions = st.number_input("Impressions",  10000, 100000, 55000, step=1000)
    clicks      = st.number_input("Clicks",       200,   15000,  3900,  step=100)
    leads       = st.number_input("Leads",        50,    9000,   1476,  step=50)
    conversions = st.number_input("Conversions",  10,    7000,   775,   step=50)
    acq_cost    = st.number_input("Acquisition Cost (₹)", 8.0, 15000.0, 376.0, step=10.0)
    engagement  = st.slider("Engagement Score", 2.5, 31.0, 13.5, step=0.1)

with col3:
    st.markdown("**Channels Used**")
    st.markdown(" ")
    email_ch  = st.checkbox("📧 Email")
    facebook  = st.checkbox("👥 Facebook")
    google    = st.checkbox("🔍 Google")
    instagram = st.checkbox("📸 Instagram")
    whatsapp  = st.checkbox("💬 WhatsApp")
    youtube   = st.checkbox("▶️ YouTube")

    st.markdown("---")
    st.markdown("**Channels selected:**")
    selected = [ch for ch,v in {
        'Email':email_ch,'Facebook':facebook,'Google':google,
        'Instagram':instagram,'WhatsApp':whatsapp,'YouTube':youtube
    }.items() if v]
    if selected:
        for ch in selected:
            st.markdown(f"✅ {ch}")
    else:
        st.caption("None selected")

inputs = dict(
    brand=brand, campaign_type=campaign_type,
    target_audience=target_audience, language=language,
    customer_segment=customer_segment, duration=duration,
    impressions=impressions, clicks=clicks, leads=leads,
    conversions=conversions, acq_cost=acq_cost,
    engagement=engagement, email=int(email_ch),
    facebook=int(facebook), google=int(google),
    instagram=int(instagram), whatsapp=int(whatsapp),
    youtube=int(youtube)
)

st.markdown("---")

# ── PREDICT BUTTON ────────────────────────────────────
predict = st.button(
    "🔮  Predict Revenue  AND  Profit/Loss",
    use_container_width=True,
    type="primary"
)

if predict:

# Input Validation
    if not selected:
        st.error("❌ Please select at least one marketing channel")
        st.stop()

    if conversions > leads:
        st.error("❌ Conversions cannot exceed Leads")
        st.stop()

    if leads > clicks:
        st.error("❌ Leads cannot exceed Clicks")
        st.stop()

    if clicks > impressions:
        st.error("❌ Clicks cannot exceed Impressions")
        st.stop()

    if acq_cost <= 0:
        st.error("❌ Acquisition Cost must be greater than 0")
        st.stop()
    with st.spinner("Running both models..."):
        errors = []

        # ── Regression ────────────────────────────────
        try:
            X_reg    = build_features(inputs, reg_pkg['features'], 'reg')
            revenue  = max(0, float(reg_pkg['model'].predict(X_reg)[0]))
            reg_ok   = True
        except Exception as e:
            reg_ok  = False
            errors.append(f"Regression error: {e}")

        # ── Classification ────────────────────────────
        try:
            X_clf    = build_features(inputs, clf_pkg['features'], 'clf')
            X_sc     = clf_pkg['scaler'].transform(X_clf)
            pred = clf_pkg['model'].predict(X_sc)[0]
            prob = clf_pkg['model'].predict_proba(X_sc)[0]
            classes = list(clf_pkg['model'].classes_)

            profit_idx = classes.index(1)
            loss_idx   = classes.index(0)

            p_profit = round(float(prob[profit_idx]) * 100, 1)
            p_loss   = round(float(prob[loss_idx]) * 100, 1)
            clf_ok   = True
        except Exception as e:
            clf_ok  = False
            errors.append(f"Classification error: {e}")

    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")

    # ── Result cards ──────────────────────────────────
    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown("### 💰 Revenue Prediction")
        if reg_ok:
            st.metric("Predicted Revenue",
                      f"₹{revenue:,.0f}",
                      f"R² = {reg_pkg['r2_score']*100:.2f}%")
            if revenue < 100000:
                st.warning("📉 Low revenue (< ₹1 Lakh)")
            elif revenue < 500000:
                st.info("📊 Average revenue (₹1L–₹5L)")
            elif revenue < 1000000:
                st.success("📈 Good revenue (₹5L–₹10L)")
            else:
                st.success("🚀 High revenue (> ₹10 Lakh)")

            roi = (revenue - acq_cost) / max(acq_cost, 1)
            st.markdown(f"**Estimated ROI** : `{roi:.2f}`")
            st.markdown(f"**Avg Model Error** : ₹{reg_pkg['rmse']:,.0f}")
        else:
            st.error("Regression failed")

    with rc2:
        st.markdown("### 📈 Profit / Loss Prediction")
        if clf_ok:
            if int(pred) == 1:
                st.metric("Prediction", "✅ PROFITABLE",
                          f"Confidence: {p_profit}%")
                st.success(f"Predicted **PROFITABLE** with **{p_profit}%** confidence")
            else:
                st.metric("Prediction", "❌ LOSS",
                          f"Confidence: {p_loss}%")
                st.error(f"Predicted **LOSS** with **{p_loss}%** confidence")

            st.markdown(f"**Profit probability** : {p_profit}%")
            st.markdown(f"**Loss probability**   : {p_loss}%")
            st.markdown(f"**Model accuracy**     : {clf_pkg['accuracy']*100:.2f}%")
        else:
            st.error("Classification failed")

    # ── Visual summary ────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Visual Summary")

    v1, v2 = st.columns(2)

    with v1:
        if reg_ok:
            # Revenue gauge
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = revenue,
                title = {'text': "Predicted Revenue (₹)",
                         'font': {'color': 'white'}},
                number= {'prefix': "₹", 'valueformat': ",.0f",
                         'font': {'color': 'white'}},
                gauge = {
                    'axis'      : {'range': [0, 2500000],
                                   'tickcolor': 'white'},
                    'bar'       : {'color': "#2ecc71"},
                    'steps'     : [
                        {'range': [0, 100000],      'color': '#c0392b'},
                        {'range': [100000, 500000],  'color': '#f39c12'},
                        {'range': [500000, 1000000], 'color': '#27ae60'},
                        {'range': [1000000, 2500000],'color': '#1abc9c'},
                    ],
                    'threshold' : {
                        'line' : {'color': 'white', 'width': 3},
                        'value': revenue
                    }
                }
            ))
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

    with v2:
        if clf_ok:
            # Probability donut
            fig = go.Figure(go.Pie(
                labels       = ['Profit ✅', 'Loss ❌'],
                values       = [p_profit, p_loss],
                hole         = 0.6,
                marker_colors= ['#2ecc71', '#e74c3c'],
                textinfo     = 'label+percent',
                hovertemplate= "%{label}: %{value}%<extra></extra>"
            ))
            fig.update_layout(
                height          = 300,
                paper_bgcolor   = 'rgba(0,0,0,0)',
                font_color      = 'white',
                showlegend      = True,
                annotations     = [{
                    'text'     : f"{'Profit' if pred==1 else 'Loss'}",
                    'x'        : 0.5, 'y': 0.5,
                    'font_size': 18,
                    'font_color': '#2ecc71' if pred==1 else '#e74c3c',
                    'showarrow': False
                }]
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Input summary ─────────────────────────────────
    st.markdown("---")
    with st.expander("📋 Full Campaign Input Summary"):
        channels = ', '.join(selected) if selected else 'None'
        summary_df = pd.DataFrame({
            'Feature'  : ['Brand','Campaign Type','Target Audience',
                          'Language','Customer Segment','Duration',
                          'Impressions','Clicks','Leads','Conversions',
                          'Acquisition Cost','Engagement Score',
                          'Channels Used'],
            'Value'    : [brand, campaign_type, target_audience,
                          language, customer_segment,
                          f"{duration} days",
                          f"{impressions:,}", f"{clicks:,}",
                          f"{leads:,}", f"{conversions:,}",
                          f"₹{acq_cost:,.2f}", engagement,
                          channels]
        })
        st.table(summary_df)

    if errors:
        for err in errors:
            st.error(err)

    # ── Combined insight ──────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Campaign Insight")
    if reg_ok and clf_ok:
        if (int(pred) == 1 and revenue >= 500000):
            st.success(f"""
**Strong Campaign** ✅
This campaign is predicted to generate **₹{revenue:,.0f}** in revenue
and is expected to be **profitable** with {p_profit}% confidence.
**Recommendation**: Proceed with this campaign strategy.
            """)
        elif int(pred) == 1 and revenue < 500000:
            st.info(f"""
**Moderate Campaign** ℹ️
This campaign is predicted to generate **₹{revenue:,.0f}** in revenue
and is expected to be **profitable** with {p_profit}% confidence.
**Recommendation**: Campaign is profitable but consider optimising
channels or targeting for higher revenue.
            """)
        else:
            st.warning(f"""
**Weak Campaign** ⚠️
This campaign is predicted to generate **₹{revenue:,.0f}** in revenue
but is expected to result in a **loss** with {p_loss}% confidence.
**Recommendation**: Review acquisition cost, targeting strategy,
or switch to higher-performing channels like Instagram or Influencer.
            """)