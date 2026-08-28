from database import init_db, PaymentFailure
from worker import execute_recovery_logic
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import uuid
import random

engine = init_db()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def run_batch_test(num_records=1000):
    db = SessionLocal()
    print(f"Generating {num_records} new PENDING failures for the batch test...")
    
    error_codes = ['ERR_INSUFFICIENT_FUNDS', 'ERR_CARD_EXPIRED', 'ERR_NETWORK', 'ERR_RISK_FLAG']
    error_probs = [0.50, 0.15, 0.25, 0.10]
    test_ids = []
    
    for _ in range(num_records):
        tx_id = str(uuid.uuid4())
        test_ids.append(tx_id)
        
        pf = PaymentFailure(
            id=tx_id,
            customer_id=f"test_batch_{random.randint(1,100)}",
            amount=round(random.uniform(99.0, 4999.0), 2),
            payment_method="credit_card",
            error_code=np.random.choice(error_codes, p=error_probs),
            failed_at=datetime.utcnow() - timedelta(days=random.randint(0, 30)),
            customer_historical_success_rate=random.uniform(0.1, 0.9),
            status="PENDING"
        )
        db.add(pf)
    
    db.commit()
    print("Executing baseline 'Dumb Executor' rules on all records...\n")
    
    # Process them (Synchronously for the test)
    for tx_id in test_ids:
        execute_recovery_logic(tx_id)
        
    # Calculate Metrics
    total_processed = len(test_ids)
    
    recovered_records = db.query(PaymentFailure).filter(
        PaymentFailure.id.in_(test_ids),
        PaymentFailure.status == 'RECOVERED'
    ).all()
    
    success_count = len(recovered_records)
    revenue_recovered = sum(r.amount for r in recovered_records)
    total_revenue_at_risk = db.query(PaymentFailure).filter(PaymentFailure.id.in_(test_ids)).with_entities(PaymentFailure.amount).all()
    total_revenue_at_risk = sum(r[0] for r in total_revenue_at_risk)
    
    print("-" * 40)
    print("BASELINE PERFORMANCE METRICS")
    print("-" * 40)
    print(f"Total Processed:      {total_processed}")
    print(f"Successful Recoveries:{success_count}")
    print(f"Recovery Rate:        {(success_count / total_processed) * 100:.2f}%")
    print(f"Revenue at Risk:      ${total_revenue_at_risk:,.2f}")
    print(f"Revenue Recovered:    ${revenue_recovered:,.2f}")
    print("-" * 40)
    
    db.close()

if __name__ == "__main__":
    import numpy as np # import here for the random choice
    np.random.seed(42)
    random.seed(42)
    run_batch_test()
