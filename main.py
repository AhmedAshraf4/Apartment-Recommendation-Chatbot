from dotenv import load_dotenv
load_dotenv()

import os
from langsmith import Client

print("API KEY PREFIX:", (os.getenv("LANGSMITH_API_KEY") or "")[:12])
print("PROJECT:", os.getenv("LANGSMITH_PROJECT"))
print("WORKSPACE:", os.getenv("LANGSMITH_WORKSPACE_ID"))

client = Client()
runs = list(client.list_runs(project_name="dorra", limit=10))
print("count returned:", len(runs))
for r in runs:
    print(r.name, r.start_time)