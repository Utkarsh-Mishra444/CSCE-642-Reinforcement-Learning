#!/usr/bin/env python3
import requests
import re
import json

def query_vgap_model(prompt):
    """Query the VGAP model via Ollama API"""
    url = "http://localhost:11434/api/chat"
    
    data = {
        "model": "vgap-2k-v1",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "top_p": 1, "num_predict": 50}
    }
    
    response = requests.post(url, json=data)
    return response.json()['message']['content']

def extract_crop_coordinates(response, original_prompt=""):
    """Extract CROP coordinates from model response using multiple strategies"""
    
    # Strategy 1: Direct CROP(x,y,x,y) format
    crop_match = re.search(r'CROP\((\d+),(\d+),(\d+),(\d+)\)', response)
    if crop_match:
        return tuple(map(int, crop_match.groups()))
    
    # Strategy 2: bbox format like bbox:(x,y,x,y)
    bbox_match = re.search(r'bbox:\((\d+),(\d+),(\d+),(\d+)\)', response)
    if bbox_match:
        return tuple(map(int, bbox_match.groups()))
    
    # Strategy 3: Look for coordinate patterns like (x=100,y=50,x2=180,y2=80)
    coord_match = re.search(r'x=(\d+),y=(\d+),x2=(\d+),y2=(\d+)', response)
    if coord_match:
        x1, y1, x2, y2 = map(int, coord_match.groups())
        return (x1, y1, x2, y2)
    
    # Strategy 4: If model suggests an element number like [1], extract bbox from original prompt
    element_match = re.search(r'\[(\d+)\]', response)
    if element_match and original_prompt:
        element_num = int(element_match.group(1))
        # Look for bbox of that element in the original prompt
        bbox_pattern = rf'\[{element_num}\].*?bbox:\((\d+),(\d+),(\d+),(\d+)\)'
        bbox_match = re.search(bbox_pattern, original_prompt, re.DOTALL)
        if bbox_match:
            return tuple(map(int, bbox_match.groups()))
    
    # Strategy 5: Look for any 4 consecutive numbers that could be coordinates
    numbers = re.findall(r'\b(\d{1,4})\b', response)
    if len(numbers) >= 4:
        # Take first 4 reasonable-sized numbers (likely coordinates)
        candidates = [int(n) for n in numbers[:8]]  # Get more candidates
        # Look for plausible coordinate pairs (x1,y1,x2,y2 where x2>x1, y2>y1)
        for i in range(len(candidates)-3):
            x1, y1, x2, y2 = candidates[i:i+4]
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 < 10000 and y2 < 10000:
                return (x1, y1, x2, y2)
    
    return None

def test_postprocessing():
    """Test post-processing on various prompts"""
    
    test_prompts = [
        # Simple search button
        """Task: Click the search button
Screen: 1280x720
Candidates:
[1] button[Search](x=850,y=15,x2=950,y2=45)
[2] input[Query](x=650,y=15,x2=840,y2=45)
[3] div[Header](x=0,y=0,x2=1280,y2=60)

Output the optimal crop region:""",
        
        # Training data format
        """Task: rent a car in Brooklyn - Central, NY on from April 9 to April 15.
Screen: 1280x5429
Candidates:
[1] <li id="bookCarTab" class="app-components-BookFlight-bookFlight__carButton--3" aria="heading level 3 Search and reserve a car"> bbox:(283,220,376,253)
[2] <div id="app"> bbox:(0,22,1280,5429)
[3] <div > bbox:(0,22,1280,5429)

Output the optimal crop region:"""
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}")
        print(f"{'='*60}")
        
        print(f"Prompt: {prompt[:100]}...")
        print()
        
        try:
            response = query_vgap_model(prompt)
            print(f"Raw Response: {response}")
            print()
            
            coords = extract_crop_coordinates(response, prompt)
            if coords:
                x1, y1, x2, y2 = coords
                print(f"✅ Extracted Coordinates: CROP({x1},{y1},{x2},{y2})")
                print(f"   Width: {x2-x1}, Height: {y2-y1}, Area: {(x2-x1)*(y2-y1)}")
                
                # Validate the crop makes sense
                if x2 <= x1 or y2 <= y1:
                    print("   ⚠️  Warning: Invalid crop (x2<=x1 or y2<=y1)")
                elif (x2-x1) > 1000 or (y2-y1) > 1000:
                    print("   ⚠️  Warning: Very large crop region")
                else:
                    print("   ✅ Valid crop region")
            else:
                print("❌ No coordinates extracted")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_postprocessing()
