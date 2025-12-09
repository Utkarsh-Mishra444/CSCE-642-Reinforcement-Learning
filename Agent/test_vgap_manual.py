#!/usr/bin/env python3
"""Manual VGAP testing script"""

import requests
import json

def test_vgap_prompt(prompt_file="vgap_test_prompt.txt"):
    """Test VGAP with a prompt file"""

    # Read the prompt
    with open(prompt_file, 'r') as f:
        prompt = f.read().strip()

    print("=== VGAP PROMPT ===")
    print(prompt)
    print("\n=== VGAP RESPONSE ===")

    # Query VGAP
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "vgap-2k-v1",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "top_p": 1, "num_predict": 200}
        }
    )

    vgap_response = response.json()['message']['content']
    print(vgap_response)

    # Try to extract coordinates
    import re
    coords = None

    # Try different patterns
    patterns = [
        r'CROP\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)',
        r'bbox[:\(]\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)',  # Allow spaces
        r'bbox:\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)',  # bbox: format with spaces
        r'x=(\d+(?:\.\d+)?),y=(\d+(?:\.\d+)?),x2=(\d+(?:\.\d+)?),y2=(\d+(?:\.\d+)?)'
    ]

    for pattern in patterns:
        match = re.search(pattern, vgap_response)
        if match:
            coords = tuple(map(float, match.groups()))
            break

    if coords:
        print(f"\n✅ EXTRACTED COORDINATES: {coords}")
    else:
        print("\n❌ NO COORDINATES FOUND")

    return vgap_response

if __name__ == "__main__":
    test_vgap_prompt()
