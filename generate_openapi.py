
import json
import sys
import os

# Add backend directory to path so imports work
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from backend.api.main import app

def generate_openapi():
    openapi_schema = app.openapi()
    with open('backend/openapi.json', 'w') as f:
        json.dump(openapi_schema, f, indent=2)
    print("Successfully generated backend/openapi.json")

if __name__ == "__main__":
    generate_openapi()
