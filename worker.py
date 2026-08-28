from huey import SqliteHuey
from database import init_db, PaymentFailure, RecoveryAudit
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import numpy as np

# Initialize Huey using a local SQLite broker
huey = SqliteHuey(filename='huey.db')

# Setup DB session
engine = init_db()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def simulate_action_outcome(record: PaymentFailure, action_taken: str, attempt_number: int) -> bool:
    """
    Replicates the environment physics (hidden rules) from the synthetic data generator.
    This allows us to deterministically test whether our chosen action succeeds.
    """
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
            base_prob = 0.20 # Mid-month blind retries mostly fail
    elif record.error_code == 'ERR_RISK_FLAG':
        base_prob = 0.10
        
    reliability_modifier = (record.customer_historical_success_rate - 0.5) * 0.2
    base_prob += reliability_modifier
    base_prob = base_prob * (0.75 ** (attempt_number - 1))
    
    final_prob = max(0.0, min(1.0, base_prob))
    return np.random.binomial(1, final_prob) == 1

def baseline_rule_engine(error_code: str) -> str:
    """
    The Dumb Executor: A traditional, non-AI rule-based approach.
    It represents how most payment gateways handle failures today.
    """
    if error_code == 'ERR_NETWORK':
        return 'silent_retry'  # Sensible for network glitches
    elif error_code == 'ERR_CARD_EXPIRED':
        return 'email_link'    # Standard cheap notification
    elif error_code == 'ERR_INSUFFICIENT_FUNDS':
        return 'silent_retry'  # Blind retry without knowing if it's payday
    else:
        return 'stop'          # Too risky or unknown

def execute_recovery_logic(transaction_id: str):
    """Core logic decoupled from Huey so we can test it synchronously in batches."""
    db = SessionLocal()
    try:
        record = db.query(PaymentFailure).filter(PaymentFailure.id == transaction_id, PaymentFailure.status == 'PENDING').first()
        if not record:
            return
            
        action = baseline_rule_engine(record.error_code)
        
        if action == 'stop':
            record.status = 'FAILED_PERMANENTLY'
            db.commit()
            return
            
        # Determine attempt number (for this baseline, we just do 1 attempt)
        attempt = 1
        
        success = simulate_action_outcome(record, action, attempt)
        
        audit = RecoveryAudit(
            failure_id=record.id,
            attempt_number=attempt,
            action_taken=action,
            outcome_success=success,
            timestamp=datetime.utcnow()
        )
        db.add(audit)
        
        if success:
            record.status = 'RECOVERED'
            record.recovered_at = datetime.utcnow()
        else:
            record.status = 'FAILED_PERMANENTLY' # Stop after 1 try for baseline
            
        db.commit()
    finally:
        db.close()

@huey.task()
def process_recovery_job(transaction_id: str):
    """Background task wrapper."""
    execute_recovery_logic(transaction_id)
