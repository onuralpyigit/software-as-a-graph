import sys
from pathlib import Path
import saag

print(saag.__all__)

# Fake JSON data for testing
import json
data = {
    "system": "test",
    "components": [{"id": "a", "type": "Application"}],
    "dependencies": []
}
with open("test_system.json", "w") as f:
    json.dump(data, f)

from saag import Pipeline

try:
    print("Testing Pipeline builder...")
    result = (
        Pipeline.from_json("test_system.json")
        .analyze(layer="system", use_ahp=True)
        .run()
    )
    print("Pipeline run successful!")
except Exception as e:
    import traceback
    traceback.print_exc()

import os
os.remove("test_system.json")
