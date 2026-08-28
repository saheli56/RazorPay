from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import uuid
from sqlalchemy.orm import Session, sessionmaker

# Import our existing database components
from database import init_db, PaymentFailure, RecoveryAudit
from worker import process_recovery_job

app = FastAPI(title="RazorRecover Ingestion API")

# Setup DB Engine
engine = init_db()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency to provide a database session for requests."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Request Validation Model using Pydantic
class PaymentFailureEvent(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float
    payment_method: str
    error_code: str
    failed_at: Optional[datetime] = None
    customer_historical_success_rate: float = 0.5 

@app.post("/api/v1/failures/ingest")
def ingest_failure(event: PaymentFailureEvent, db: Session = Depends(get_db)):
    """
    Ingests a new payment failure event.
    """
    # 1. Idempotency Check: Don't process the same webhook twice.
    existing_failure = db.query(PaymentFailure).filter(PaymentFailure.id == event.transaction_id).first()
    if existing_failure:
        # Webhooks often retry on 5xx or 4xx, so a 200 OK ensures the gateway stops retrying this event.
        return {"status": "success", "message": "Event already processed", "transaction_id": event.transaction_id}
    
    event_time = event.failed_at or datetime.utcnow()
    
    # 2. Create core failure record in a 'PENDING' state waiting for AI action
    new_failure = PaymentFailure(
        id=event.transaction_id,
        customer_id=event.customer_id,
        amount=event.amount,
        payment_method=event.payment_method,
        error_code=event.error_code,
        failed_at=event_time,
        customer_historical_success_rate=event.customer_historical_success_rate,
        status="PENDING"
    )
    
    # 3. Create initial audit entry logging the ingestion itself
    initial_audit = RecoveryAudit(
        failure_id=event.transaction_id,
        attempt_number=0,
        action_taken="system_ingestion",
        outcome_success=False, # Hasn't succeeded yet
        timestamp=event_time
    )
    
    db.add(new_failure)
    db.add(initial_audit)
    
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
    # Queue the background recovery job
    process_recovery_job(event.transaction_id)
        
    return {"status": "success", "message": "Failure event ingested and job queued", "transaction_id": event.transaction_id}

if __name__ == "__main__":
    # Test Block to run and verify the API
    import uvicorn
    import threading
    import time
    import requests
    
    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="critical")
        
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    time.sleep(2) # Give server time to boot
    
    test_id = str(uuid.uuid4())
    test_event = {
        "transaction_id": test_id,
        "customer_id": "cust_test_api_001",
        "amount": 1999.0,
        "payment_method": "credit_card",
        "error_code": "ERR_INSUFFICIENT_FUNDS",
        "customer_historical_success_rate": 0.85
    }
    
    print("--- Testing Ingestion Endpoint ---")
    
    # Test 1: New Event
    response = requests.post("http://127.0.0.1:8000/api/v1/failures/ingest", json=test_event)
    print("\nTest 1 (New Event):")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 2: Idempotency (Duplicate Event)
    response_dup = requests.post("http://127.0.0.1:8000/api/v1/failures/ingest", json=test_event)
    print("\nTest 2 (Idempotency - Duplicate Event):")
    print(f"Status Code: {response_dup.status_code}")
    print(f"Response: {response_dup.json()}")
    
    # Verify Database State
    db = SessionLocal()
    record = db.query(PaymentFailure).filter(PaymentFailure.id == test_id).first()
    audit_count = db.query(RecoveryAudit).filter(RecoveryAudit.failure_id == test_id).count()
    print("\nTest 3 (Database Verification):")
    print(f"Record Status: {record.status if record else 'Missing'}")
    print(f"Initial Audit Entries Created: {audit_count}")
