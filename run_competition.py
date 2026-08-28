from database import init_db, PaymentFailure, RecoveryAudit
from worker import execute_recovery_logic, ACTION_COSTS
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import uuid
import random
import numpy as np
import json

engine = init_db()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def run_ab_test():
    db = SessionLocal()
    
    # 1. Generate 1000 base records with a fixed seed
    np.random.seed(999)
    random.seed(999)
    
    print("Generating 1,000 fresh payment failures for A/B Testing...")
    error_codes = ['ERR_INSUFFICIENT_FUNDS', 'ERR_CARD_EXPIRED', 'ERR_NETWORK', 'ERR_RISK_FLAG']
    error_probs = [0.50, 0.15, 0.25, 0.10]
    
    base_records = []
    for _ in range(1000):
        failed_at = datetime.utcnow() - timedelta(days=random.randint(0, 30))
        # Ensure we hit some end-of-month dates for the AI to demonstrate its temporal learning
        if random.random() < 0.2:
            failed_at = failed_at.replace(day=random.choice([28, 29, 30, 31]))
            
        base_records.append({
            'orig_id': str(uuid.uuid4()),
            'customer_id': f"ab_test_{random.randint(1,100)}",
            'amount': round(random.uniform(99.0, 4999.0), 2),
            'payment_method': "credit_card",
            'error_code': np.random.choice(error_codes, p=error_probs),
            'failed_at': failed_at,
            'customer_historical_success_rate': random.uniform(0.1, 0.9)
        })
        
    baseline_ids = []
    ai_ids = []
    
    # 2. Insert duplicates for Baseline and AI to ensure no DB collision
    for rec in base_records:
        b_id = rec['orig_id'] + '-baseline'
        a_id = rec['orig_id'] + '-ai'
        
        baseline_ids.append(b_id)
        ai_ids.append(a_id)
        
        db.add(PaymentFailure(id=b_id, customer_id=rec['customer_id'], amount=rec['amount'], payment_method=rec['payment_method'], error_code=rec['error_code'], failed_at=rec['failed_at'], customer_historical_success_rate=rec['customer_historical_success_rate'], status='PENDING'))
        db.add(PaymentFailure(id=a_id, customer_id=rec['customer_id'], amount=rec['amount'], payment_method=rec['payment_method'], error_code=rec['error_code'], failed_at=rec['failed_at'], customer_historical_success_rate=rec['customer_historical_success_rate'], status='PENDING'))
        
    db.commit()
    
    # 3. Execute Strategies
    print("Executing Baseline Strategy (The Dumb Executor)...")
    for bid in baseline_ids:
        execute_recovery_logic(bid, engine_type='baseline')
        
    print("Executing AI Decision Engine Strategy...")
    for aid in ai_ids:
        execute_recovery_logic(aid, engine_type='ai')
        
    # 4. Calculate Metrics
    def calc_metrics(test_ids):
        recovered = db.query(PaymentFailure).filter(PaymentFailure.id.in_(test_ids), PaymentFailure.status == 'RECOVERED').all()
        audits = db.query(RecoveryAudit).filter(RecoveryAudit.failure_id.in_(test_ids)).all()
        
        gross_rev = sum(r.amount for r in recovered)
        action_cost = sum(ACTION_COSTS.get(a.action_taken, 0) for a in audits)
        net_rev = gross_rev - action_cost
        
        return {
            'processed': len(test_ids),
            'recovered': len(recovered),
            'rate': len(recovered) / len(test_ids),
            'gross': gross_rev,
            'cost': action_cost,
            'net': net_rev
        }
        
    b_metrics = calc_metrics(baseline_ids)
    a_metrics = calc_metrics(ai_ids)
    
    print("\n" + "="*60)
    print("        RAZORRECOVER AI vs BASELINE COMPETITION        ")
    print("="*60)
    
    print(f"{'Metric':<25} | {'Baseline':<15} | {'AI Engine':<15}")
    print("-" * 60)
    print(f"{'Total Processed':<25} | {b_metrics['processed']:<15} | {a_metrics['processed']:<15}")
    print(f"{'Successful Recoveries':<25} | {b_metrics['recovered']:<15} | {a_metrics['recovered']:<15}")
    print(f"{'Recovery Rate':<25} | {b_metrics['rate']:<14.2%} | {a_metrics['rate']:<14.2%}")
    print(f"{'Gross Revenue':<25} | ${b_metrics['gross']:,.2f} | ${a_metrics['gross']:,.2f}")
    print(f"{'Action Costs':<25} | ${b_metrics['cost']:,.2f} | ${a_metrics['cost']:,.2f}")
    print(f"{'Net Revenue':<25} | ${b_metrics['net']:,.2f} | ${a_metrics['net']:,.2f}")
    print("=" * 60)
    
    rate_diff = a_metrics['rate'] - b_metrics['rate']
    net_diff = a_metrics['net'] - b_metrics['net']
    print(f" AI Improvement (Rate): +{rate_diff:.2%}")
    print(f" AI Improvement (Net):  +${net_diff:,.2f}")
    
    # 5. Show examples of AI acting differently
    print("\n--- Example Decisions (Where AI Outsmarted Baseline) ---")
    ai_success_ids = [r.id.replace('-ai', '') for r in db.query(PaymentFailure).filter(PaymentFailure.id.in_(ai_ids), PaymentFailure.status == 'RECOVERED').all()]
    base_fail_ids = [r.id.replace('-baseline', '') for r in db.query(PaymentFailure).filter(PaymentFailure.id.in_(baseline_ids), PaymentFailure.status == 'FAILED_PERMANENTLY').all()]
    
    better_ids = list(set(ai_success_ids) & set(base_fail_ids))
    
    for i in range(min(3, len(better_ids))):
        orig = better_ids[i]
        b_rec = db.query(PaymentFailure).filter(PaymentFailure.id == orig+'-baseline').first()
        a_rec = db.query(PaymentFailure).filter(PaymentFailure.id == orig+'-ai').first()
        
        b_aud = db.query(RecoveryAudit).filter(RecoveryAudit.failure_id == orig+'-baseline').first()
        a_aud = db.query(RecoveryAudit).filter(RecoveryAudit.failure_id == orig+'-ai').first()
        
        print(f"\n[Case {i+1}] Error: {b_rec.error_code} | Amount: ${b_rec.amount:,.2f} | Failed On: {b_rec.failed_at.date()} (Day {b_rec.failed_at.day})")
        print(f"  Baseline Chose: {b_aud.action_taken} -> FAILED")
        print(f"  AI Chose:       {a_aud.action_taken} -> RECOVERED")
        try:
            evs = json.loads(a_aud.ai_predicted_action)
            print("  AI Internal EV Calculation:")
            for k,v in evs.items():
                print(f"    - {k}: Prob={v['prob']:.1%}, Cost=${v['cost']:.2f}, EV=${v['ev']:.2f}")
        except:
            pass

    db.close()

if __name__ == '__main__':
    run_ab_test()
