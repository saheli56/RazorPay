
from huey import SqliteHuey
from database import init_db, PaymentFailure
from sqlalchemy.orm import sessionmaker

# 1. Initialize Huey using a local SQLite broker for hackathon simplicity
huey = SqliteHuey(filename='huey.db')

# Setup DB session for the worker
engine = init_db()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@huey.task()
def process_recovery_job(transaction_id: str):
    """
    Background task that receives a failed payment and processes the recovery strategy.
    """
    print(f"\n[WORKER] Received recovery job for transaction: {transaction_id}")
    
    db = SessionLocal()
    try:
        record = db.query(PaymentFailure).filter(PaymentFailure.id == transaction_id).first()
        if not record:
            print(f"[WORKER] Error: Transaction {transaction_id} not found in DB.")
            return
            
        print(f"[WORKER] Successfully loaded PENDING record: {record.id}")
        print(f"[WORKER] Details: {record.payment_method} - {record.error_code} - ${record.amount}")
        print(f"[WORKER] (Actual recovery logic/decision will be implemented here next)")
        
        # Note: We are deliberately NOT updating the final outcome or status yet as per instructions.
        
    finally:
        db.close()
