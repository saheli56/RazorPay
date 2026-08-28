import pandas as pd
from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import uuid
import os

Base = declarative_base()

# ---------------------------------------------------------
# DATABASE SCHEMA
# ---------------------------------------------------------

class PaymentFailure(Base):
    """
    Represents the core payment failure event ingested from the payment gateway.
    """
    __tablename__ = 'payment_failures'
    
    id = Column(String, primary_key=True) # maps to transaction_id
    customer_id = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    payment_method = Column(String, nullable=False)
    error_code = Column(String, nullable=False)
    failed_at = Column(DateTime, nullable=False)
    
    # Customer context for ML
    customer_historical_success_rate = Column(Float, nullable=False)
    
    # Track the current state of this failure lifecycle
    status = Column(String, nullable=False, default='PENDING') # 'PENDING', 'RECOVERED', 'FAILED_PERMANENTLY'
    recovered_at = Column(DateTime, nullable=True)
    
    # Relationship to audit trail
    audits = relationship("RecoveryAudit", back_populates="failure", cascade="all, delete-orphan")

class RecoveryAudit(Base):
    """
    An append-only audit log of every action taken to recover a payment failure.
    Extensible for AI predictions and tracking multiple retry attempts.
    """
    __tablename__ = 'recovery_audits'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    failure_id = Column(String, ForeignKey('payment_failures.id'), nullable=False)
    
    attempt_number = Column(Integer, nullable=False)
    action_taken = Column(String, nullable=False)
    
    # AI tracking (for later phases)
    ai_predicted_action = Column(String, nullable=True)
    ai_confidence = Column(Float, nullable=True)
    
    outcome_success = Column(Boolean, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    failure = relationship("PaymentFailure", back_populates="audits")

# ---------------------------------------------------------
# DATABASE OPERATIONS
# ---------------------------------------------------------

def init_db(db_url="sqlite:///razorrecover.db"):
    """Initialize the database engine and create tables."""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return engine

def load_csv_data(engine, csv_path="payment_failures.csv"):
    """Load the generated CSV data into the normalized database schema."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    print("Reading CSV data...")
    df = pd.read_csv(csv_path)
    
    # Parse dates (Handling NaNs for recovered_at)
    df['failed_at'] = pd.to_datetime(df['failed_at'])
    df['recovered_at'] = pd.to_datetime(df['recovered_at'])
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Clear existing data for idempotency during hackathon development
    session.query(RecoveryAudit).delete()
    session.query(PaymentFailure).delete()
    session.commit()
    
    failures_to_insert = []
    audits_to_insert = []
    
    print("Transforming and inserting records...")
    for _, row in df.iterrows():
        # Determine status
        status = 'RECOVERED' if row['recovery_success'] else 'FAILED_PERMANENTLY'
        
        # 1. Create the core Failure Event
        pf = PaymentFailure(
            id=row['transaction_id'],
            customer_id=row['customer_id'],
            amount=row['amount'],
            payment_method=row['payment_method'],
            error_code=row['error_code'],
            failed_at=row['failed_at'].to_pydatetime(),
            customer_historical_success_rate=row['customer_historical_success_rate'],
            status=status,
            recovered_at=row['recovered_at'].to_pydatetime() if pd.notnull(row['recovered_at']) else None
        )
        
        # 2. Create the Audit/Action Trail
        # If it was recovered, we stamp the audit at recovery time. If it failed, we simulate the action happened shortly after failure.
        audit_time = row['recovered_at'].to_pydatetime() if pd.notnull(row['recovered_at']) else (row['failed_at'] + pd.Timedelta(hours=2)).to_pydatetime()
        
        ra = RecoveryAudit(
            failure_id=row['transaction_id'],
            attempt_number=row['retry_attempt_number'],
            action_taken=row['recovery_action_taken'],
            outcome_success=row['recovery_success'],
            timestamp=audit_time
        )
        
        failures_to_insert.append(pf)
        audits_to_insert.append(ra)
        
    # Bulk insert for speed
    session.bulk_save_objects(failures_to_insert)
    session.bulk_save_objects(audits_to_insert)
    session.commit()
    print(f"Successfully loaded {len(failures_to_insert)} failures and {len(audits_to_insert)} audits into the database.")

def verify_data(engine):
    """Retrieve and verify a few records to ensure integrity."""
    Session = sessionmaker(bind=engine)
    session = Session()
    
    pf_count = session.query(PaymentFailure).count()
    ra_count = session.query(RecoveryAudit).count()
    
    print("\n--- Database Verification ---")
    print(f"Total Payment Failures: {pf_count}")
    print(f"Total Recovery Audits: {ra_count}")
    
    # Grab one recovered record with its audit trail
    sample = session.query(PaymentFailure).filter_by(status='RECOVERED').first()
    if sample:
        print("\nSample Recovered Record:")
        print(f"  Transaction ID: {sample.id}")
        print(f"  Error Code: {sample.error_code}")
        print(f"  Status: {sample.status}")
        
        audit = sample.audits[0]
        print("  Audit Trail:")
        print(f"    - Attempt {audit.attempt_number}: {audit.action_taken} -> Success: {audit.outcome_success}")

if __name__ == '__main__':
    db_engine = init_db()
    load_csv_data(db_engine)
    verify_data(db_engine)
