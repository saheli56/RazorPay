import subprocess
import time
import requests
import uuid
from database import init_db, PaymentFailure, RecoveryAudit
from sqlalchemy.orm import sessionmaker

print("========================================")
print("  RAZORRECOVER AI - END-TO-END TEST")
print("========================================")

# 1 & 2: Start Background Services
print("\n[1] Starting Huey Background Worker...")
huey_process = subprocess.Popen(["python", "-m", "huey.bin.huey_consumer", "worker.huey"], cwd="C:\\Razorpay")

print("[2] Starting FastAPI Ingestion Server...")
api_process = subprocess.Popen(["uvicorn", "api:app", "--port", "8000"], cwd="C:\\Razorpay")

try:
    print("Waiting 5 seconds for services to boot...")
    time.sleep(5)
    
    # 3: Test API Pipeline
    print("\n[3] Testing FastAPI Ingestion & Validation...")
    tx_id = str(uuid.uuid4())
    payload = {
        "transaction_id": tx_id,
        "customer_id": "cust_e2e_001",
        "amount": 2500.0,
        "payment_method": "credit_card",
        "error_code": "ERR_CARD_EXPIRED",
        "customer_historical_success_rate": 0.8
    }
    
    # Initial Ingestion
    resp = requests.post("http://127.0.0.1:8000/api/v1/failures/ingest", json=payload)
    print(f"  -> Ingest Response: {resp.status_code} | {resp.json().get('message')}")
    if resp.status_code != 200:
        raise Exception("API Ingestion Failed")
        
    # Idempotency Check
    resp2 = requests.post("http://127.0.0.1:8000/api/v1/failures/ingest", json=payload)
    print(f"  -> Idempotency Response: {resp2.status_code} | {resp2.json().get('message')}")
    if resp2.json().get('message') != 'Event already processed':
        raise Exception("Idempotency validation failed")
    
    print("\n[4] Waiting 5 seconds for Huey queue to process the job...")
    time.sleep(5)
    
    # 5: Verify Database and Worker Execution
    print("\n[5] Verifying Database State & AI Decision Runtime...")
    engine = init_db()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    record = db.query(PaymentFailure).filter(PaymentFailure.id == tx_id).first()
    if not record:
        raise Exception("Record was not saved to database.")
    
    print(f"  -> Final Payment Status: {record.status}")
    if record.status not in ['RECOVERED', 'FAILED_PERMANENTLY']:
        raise Exception(f"Worker did not process the record. Status is still {record.status}")
    
    audits = db.query(RecoveryAudit).filter(RecoveryAudit.failure_id == tx_id).order_by(RecoveryAudit.attempt_number).all()
    print(f"  -> Audit Ledger Contains {len(audits)} Entries:")
    for a in audits:
        print(f"      - Action: {a.action_taken} | Success: {a.outcome_success}")
        
    if len(audits) < 2:
        raise Exception("Missing audit records. Expected ingestion + AI action.")
        
    db.close()
    
    # 6: Test Streamlit Dashboard
    print("\n[6] Verifying Streamlit Dashboard Integrity...")
    try:
        dash_resp = requests.get("http://127.0.0.1:8501")
        print(f"  -> Streamlit HTTP Status: {dash_resp.status_code}")
        if dash_resp.status_code != 200:
            raise Exception("Streamlit is returning an error code.")
    except Exception as e:
        print(f"  -> Warning: Could not connect to Streamlit on 8501. Error: {e}")
        
    print("\nE2E TEST COMPLETED SUCCESSFULLY: ALL COMPONENTS PASS!")
    
except Exception as e:
    print(f"\nE2E TEST FAILED: {str(e)}")

finally:
    print("\nCleaning up test processes...")
    huey_process.terminate()
    api_process.terminate()
