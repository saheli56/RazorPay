import streamlit as st
import pandas as pd
from database import init_db
from sqlalchemy.orm import sessionmaker

# 1. Page Configuration (Must be first Streamlit command)
st.set_page_config(page_title="RazorRecover AI", page_icon="", layout="wide")

# 2. Custom CSS for a polished, Hackathon-winning look
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 25px;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .metric-value { font-size: 2.2rem; font-weight: 800; color: #1e1e1e; font-family: monospace; }
    .metric-label { font-size: 0.9rem; color: #6c757d; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px;}
    .uplift-card {
        background: linear-gradient(135deg, #f0fff4 0%, #d4edda 100%);
        border: 2px solid #28a745;
    }
    .uplift-value { color: #155724; font-size: 2.5rem; font-weight: 900; font-family: monospace;}
    .uplift-percent { color: #155724; font-weight: bold; font-size: 1.1rem; }
    .header-text { color: #4a4a4a; font-size: 1.2rem; margin-bottom: 30px;}
    </style>
""", unsafe_allow_html=True)

# 3. Data Loader
@st.cache_data(ttl=60)
def load_kpi_data():
    """Connects to the SQLite DB and pulls real project data."""
    engine = init_db()
    
    # Query failures and join with their audits
    # We filter to records that were part of the A/B test batch
    query = """
    SELECT 
        pf.id, pf.amount, pf.status, 
        ra.action_taken, ra.ai_predicted_action
    FROM payment_failures pf
    JOIN recovery_audits ra ON pf.id = ra.failure_id
    WHERE pf.id LIKE '%-baseline' OR pf.id LIKE '%-ai'
    """
    df = pd.read_sql_query(query, engine)
    
    if df.empty:
        return None
        
    # EXPLICIT STRATEGY IDENTIFICATION
    # As requested, we don't just rely on the ID suffix. 
    # We inspect the actual audit payload: the Baseline always writes "{}" (empty JSON) for AI predictions,
    # whereas the AI Engine writes a populated JSON dictionary of its Expected Value calculations.
    df['strategy'] = df['ai_predicted_action'].apply(lambda x: 'Baseline' if x == '{}' or pd.isna(x) else 'AI Engine')
    
    # Calculate costs and net revenue dynamically
    ACTION_COSTS = {
        'silent_retry': 0.10,
        'email_link': 0.05,
        'whatsapp_link': 0.50,
        'stop': 0.00
    }
    
    df['cost'] = df['action_taken'].map(ACTION_COSTS).fillna(0)
    df['recovered_amount'] = df.apply(lambda row: row['amount'] if row['status'] == 'RECOVERED' else 0, axis=1)
    df['net_revenue'] = df['recovered_amount'] - df['cost']
    
    return df

# 4. UI Rendering Functions
def render_header():
    st.title("RazorRecover AI")
    st.markdown("""
    <div class="header-text">
        An intelligent revenue recovery engine that replaces rigid, rule-based retry logic with a dynamic Expected Value (EV) engine. <br>
        <strong>It evaluates the exact probability of success for every communication channel and executes the intervention mathematically proven to maximize Net Revenue.</strong>
    </div>
    """, unsafe_allow_html=True)

def render_hero_kpis():
    df = load_kpi_data()
    
    if df is None or df.empty:
        st.warning("⚠️ No A/B test data found in the database. Please run the competition script to populate data.")
        return
        
    # Aggregate Metrics
    baseline_df = df[df['strategy'] == 'Baseline']
    ai_df = df[df['strategy'] == 'AI Engine']
    
    # Total at risk is the sum of original failure amounts. 
    # Since Baseline processed all of them, we can sum the baseline amounts.
    total_at_risk = baseline_df['amount'].sum()
    
    baseline_net = baseline_df['net_revenue'].sum()
    ai_net = ai_df['net_revenue'].sum()
    
    uplift = ai_net - baseline_net
    uplift_percent = (uplift / baseline_net) * 100 if baseline_net > 0 else 0
    
    # Render KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Revenue at Risk</div>
            <div class="metric-value">${total_at_risk:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Legacy Baseline Net</div>
            <div class="metric-value">${baseline_net:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RazorRecover AI Net</div>
            <div class="metric-value" style="color: #28a745;">${ai_net:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card uplift-card">
            <div class="metric-label" style="color: #155724;">AI Net Revenue Uplift</div>
            <div class="uplift-value">+${uplift:,.0f}</div>
            <div class="uplift-percent">(+{uplift_percent:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    render_header()
    render_hero_kpis()
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'><em>Additional charts and Audit Trail explorer will go here...</em></p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
