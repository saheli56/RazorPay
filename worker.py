import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from database import init_db, PaymentFailure, RecoveryAudit
from sqlalchemy.orm import sessionmaker
from huey import SqliteHuey

huey = SqliteHuey(filename='huey.db')
engine = init_db()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Load AI Model Globally
try:
    AI_MODEL = joblib.load("recovery_model.joblib")
except Exception as e:
    print(f"Warning: Could not load AI model. {e}")
    AI_MODEL = None

ACTION_COSTS = {
    'silent_retry': 0.10,   # API call cost
    'email_link': 0.05,     # Email provider cost
    'whatsapp_link': 0.50,  # Twilio/WhatsApp API cost
    'stop': 0.00
}

def simulate_action_outcome(record: PaymentFailure, action_taken: str, attempt_number: int) -> bool:
    """Deterministic environment physics simulator for A/B testing."""
    if action_taken == 'stop':
        return False
        
    base_prob = 0.0
    if record.error_code == 'ERR_CARD_EXPIRED':
        base_prob = 0.01 if action_taken == 'silent_retry' else 0.45
    elif record.error_code == 'ERR_NETWORK':
        base_prob = 0.85 if action_taken == 'silent_retry' else 0.60
    elif record.error_code == 'ERR_INSUFFICIENT_FUNDS':
        day_of_month = record.failed_at.day
        if day_of_month >= 25 and action_taken == 'silent_retry':
            base_prob = 0.75
        elif action_taken == 'whatsapp_link':
            base_prob = 0.40
        else:
            base_prob = 0.20
    elif record.error_code == 'ERR_RISK_FLAG':
        base_prob = 0.10
        
    reliability_modifier = (record.customer_historical_success_rate - 0.5) * 0.2
    base_prob += reliability_modifier
    base_prob = base_prob * (0.75 ** (attempt_number - 1))
    final_prob = max(0.0, min(1.0, base_prob))
    
    # Crucial: Fix the seed based on the core ID so identical actions produce identical outcomes across A/B tests
    orig_id = record.id.replace('-baseline', '').replace('-ai', '')
    seed_str = f"{orig_id}_{action_taken}_{attempt_number}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32 - 1)
    np.random.seed(seed)
    
    return np.random.binomial(1, final_prob) == 1

def baseline_rule_engine(record: PaymentFailure) -> tuple:
    action = 'stop'
    if record.error_code == 'ERR_NETWORK':
        action = 'silent_retry'
    elif record.error_code == 'ERR_CARD_EXPIRED':
        action = 'email_link'
    elif record.error_code == 'ERR_INSUFFICIENT_FUNDS':
        action = 'silent_retry'
        
    return action, 0.0, "{}"

def ai_decision_engine(record: PaymentFailure) -> tuple:
    if not AI_MODEL:
        return 'stop', 0.0, "{}"
        
    # 2. Extract features safe at decision time
    base_features = {
        'amount': record.amount,
        'payment_method': record.payment_method,
        'error_code': record.error_code,
        'customer_historical_success_rate': record.customer_historical_success_rate,
        'retry_attempt_number': 1,
        'day_of_month': record.failed_at.day,
        'day_of_week': record.failed_at.weekday(),
        'hour_of_day': record.failed_at.hour
    }
    
    # 3. Create candidate inputs for valid actions
    candidates = []
    actions = ['silent_retry', 'email_link', 'whatsapp_link']
    for action in actions:
        row = base_features.copy()
        row['recovery_action_taken'] = action
        candidates.append(row)
        
    df_candidates = pd.DataFrame(candidates)
    
    # 4. Predict P(success)
    probs = AI_MODEL.predict_proba(df_candidates)[:, 1]
    
    best_action = 'stop'
    best_ev = 0.0 # Threshold for taking any action
    best_prob = 0.0
    evaluations = {}
    
    # 5 & 6 & 7. Calculate EV and pick best
    for i, action in enumerate(actions):
        p = float(probs[i])
        cost = ACTION_COSTS[action]
        ev = (p * record.amount) - cost
        evaluations[action] = {'prob': round(p, 4), 'cost': cost, 'ev': round(ev, 2)}
        
        if ev > best_ev:
            best_ev = ev
            best_action = action
            best_prob = p
            
    return best_action, best_prob, json.dumps(evaluations)

def execute_recovery_logic(transaction_id: str, engine_type: str = 'ai'):
    db = SessionLocal()
    try:
        record = db.query(PaymentFailure).filter(PaymentFailure.id == transaction_id, PaymentFailure.status == 'PENDING').first()
        if not record:
            return
            
        if engine_type == 'ai':
            action, prob, eval_json = ai_decision_engine(record)
        else:
            action, prob, eval_json = baseline_rule_engine(record)
            
        if action == 'stop':
            record.status = 'FAILED_PERMANENTLY'
            audit = RecoveryAudit(
                failure_id=record.id,
                attempt_number=1,
                action_taken='stop',
                outcome_success=False,
                timestamp=datetime.utcnow(),
                ai_predicted_action=eval_json,
                ai_confidence=prob
            )
            db.add(audit)
            db.commit()
            return
            
        attempt = 1
        success = simulate_action_outcome(record, action, attempt)
        
        audit = RecoveryAudit(
            failure_id=record.id,
            attempt_number=attempt,
            action_taken=action,
            outcome_success=success,
            timestamp=datetime.utcnow(),
            ai_predicted_action=eval_json,
            ai_confidence=prob
        )
        db.add(audit)
        
        if success:
            record.status = 'RECOVERED'
            record.recovered_at = datetime.utcnow()
        else:
            record.status = 'FAILED_PERMANENTLY'
            
        db.commit()
    finally:
        db.close()

@huey.task()
def process_recovery_job(transaction_id: str):
    execute_recovery_logic(transaction_id, engine_type='ai')
