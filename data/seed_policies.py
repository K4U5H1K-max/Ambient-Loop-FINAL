"""
Seed PostgreSQL database with policies defined in policies.py
"""
import site
from pathlib import Path

site.addsitedir(str(Path(__file__).parent.parent))

from data.ticket_db import Policy
from data.ticket_db import get_session
from data.policies import get_all_policies

def seed_policies_from_py():
    session = get_session()
    
    # Check if already seeded
    existing_count = session.query(Policy).count()
    if existing_count > 0:
        print(f"↩️ Policies already seeded ({existing_count} policies exist). Skipping.")
        session.close()
        return
    
    policies = get_all_policies()

    for name, details in policies.items():
        existing = session.query(Policy).filter_by(policy_name=name).first()
        if not existing:
            new_policy = Policy(
                policy_name=name,
                description=details["description"],
                when_to_use=details["when_to_use"],
                applicable_problems=details["applicable_problems"]
            )
            session.add(new_policy)
            print(f"✅ Added policy: {name}")
        else:
            print(f"↩️ Policy already exists: {name}")

    session.commit()
    session.close()
    print("🎉 Policy table seeded successfully!")

if __name__ == "__main__":
    seed_policies_from_py()