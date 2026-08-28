
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta

def generate_synthetic_data(num_records=10000, output_file="payment_failures.csv"):
    """
    Generates a synthetic dataset of payment failures with hidden, discoverable 
    patterns for an AI Revenue Recovery ML model.
    """
    np.random.seed(42)
    
    # 1. Base Entities
    customer_ids = [f"cust_{i}" for i in range(1, (num_records // 5) + 1)]
    error_codes = ['ERR_INSUFFICIENT_FUNDS', 'ERR_CARD_EXPIRED', 'ERR_NETWORK', 'ERR_RISK_FLAG']
    error_probs = [0.50, 0.15, 0.25, 0.10]
    
    payment_methods = ['credit_card', 'debit_card', 'upi', 'enach']
    recovery_actions = ['silent_retry', 'whatsapp_link', 'email_link']
    
    # Generate base data
    data = []
    
    # Base timestamp generation (last 6 months)
    start_date = datetime.now() - timedelta(days=180)
    
    for _ in range(num_records):
        cust_id = np.random.choice(customer_ids)
        # Assign a historical success rate to the customer (beta distribution skewed towards high success)
        hist_success_rate = np.random.beta(a=8, b=2) 
        
        failed_at = start_date + timedelta(days=np.random.randint(0, 180), hours=np.random.randint(0, 24))
        error_code = np.random.choice(error_codes, p=error_probs)
        amount = np.round(np.random.uniform(99.0, 4999.0), 2)
        payment_method = np.random.choice(payment_methods)
        
        # Determine the action taken (randomly assigned for the training set to allow exploration)
        action_taken = np.random.choice(recovery_actions)
        retry_attempt = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        
        # --- HIDDEN PATTERNS FOR ML DISCOVERY ---
        base_prob = 0.0
        
        # Pattern 1: Card Expired requires user intervention
        if error_code == 'ERR_CARD_EXPIRED':
            if action_taken == 'silent_retry':
                base_prob = 0.01  # Near zero
            else:
                base_prob = 0.45  # WhatsApp/Email has a chance
                
        # Pattern 2: Network errors are highly recoverable via silent retries
        elif error_code == 'ERR_NETWORK':
            if action_taken == 'silent_retry':
                base_prob = 0.85
            else:
                base_prob = 0.60  # Bothering user for network error drops conversion
                
        # Pattern 3: Insufficient Funds & Salary Day Effect (End of month failures)
        elif error_code == 'ERR_INSUFFICIENT_FUNDS':
            day_of_month = failed_at.day
            if day_of_month >= 25 and action_taken == 'silent_retry':
                # Assuming retry happens early next month (salary day)
                base_prob = 0.75 
            elif action_taken == 'whatsapp_link':
                base_prob = 0.40
            else:
                base_prob = 0.20
                
        # Pattern 4: Risk Flags are hard to recover
        elif error_code == 'ERR_RISK_FLAG':
            base_prob = 0.10
            
        # Pattern 5: Customer Reliability (higher historical success = higher recovery chance)
        # Shift probability slightly based on historical reliability
        reliability_modifier = (hist_success_rate - 0.5) * 0.2
        base_prob += reliability_modifier
        
        # Pattern 6: Retry Fatigue (probability drops with more attempts)
        base_prob = base_prob * (0.75 ** (retry_attempt - 1))
        
        # Clamp probability
        final_prob = max(0.0, min(1.0, base_prob))
        
        # Roll the dice to determine actual success
        recovery_success = np.random.binomial(1, final_prob)
        
        recovered_at = None
        if recovery_success:
            # If successful, assign a recovery timestamp
            hours_to_recover = np.random.randint(1, 72)
            recovered_at = failed_at + timedelta(hours=hours_to_recover)
            
        data.append({
            "transaction_id": str(uuid.uuid4()),
            "customer_id": cust_id,
            "failed_at": failed_at,
            "amount": amount,
            "payment_method": payment_method,
            "error_code": error_code,
            "customer_historical_success_rate": round(hist_success_rate, 3),
            "retry_attempt_number": retry_attempt,
            "recovery_action_taken": action_taken,
            "recovery_success": bool(recovery_success),
            "recovered_at": recovered_at,
            "hidden_true_probability": round(final_prob, 3) # Included for debug/validation, drop before training
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    print(f"Successfully generated {num_records} records and saved to {output_file}")
    
    # Print validation statistics
    print("\n--- Validation Statistics ---")
    print(f"Overall Recovery Rate: {df['recovery_success'].mean():.2%}")
    print("\nRecovery Rate by Error Code:")
    print(df.groupby('error_code')['recovery_success'].mean())
    print("\nRecovery Rate by Action Taken (for Card Expired):")
    print(df[df['error_code'] == 'ERR_CARD_EXPIRED'].groupby('recovery_action_taken')['recovery_success'].mean())
    print("\nRecovery Rate by Action Taken (for Network Error):")
    print(df[df['error_code'] == 'ERR_NETWORK'].groupby('recovery_action_taken')['recovery_success'].mean())
    
    return df

if __name__ == "__main__":
    generate_synthetic_data()
