from litellm import completion
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

print("=== ENV CHECK ===")
print("OPENROUTER_API_KEY exists:", os.getenv("OPENROUTER_API_KEY") is not None)
print("OPENROUTER_API_KEY prefix:", os.getenv("OPENROUTER_API_KEY")[:10])

try:
    response = completion(
        model="openrouter/nvidia/nemotron-3-super-120b-a12b:free",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        messages=[
            {
                "role": "user",
                "content": "Say hello in one short sentence."
            }
        ]
    )

    print("\n=== SUCCESS ===")
    print(response)

except Exception as e:
    print("\n=== ERROR ===")
    print(type(e))
    print(e)