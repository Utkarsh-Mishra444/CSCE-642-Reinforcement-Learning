"""
Download and verify Multimodal-Mind2Web dataset.

This script:
1. Downloads the dataset from HuggingFace (streaming mode to handle 13.6GB)
2. Verifies the data structure and bbox field access
3. Saves sample data for local testing
"""

import json
import base64
from io import BytesIO
from pathlib import Path
from datasets import load_dataset
from PIL import Image


def decode_screenshot(screenshot_base64: str) -> Image.Image:
    """Decode base64 screenshot to PIL Image."""
    image_data = base64.b64decode(screenshot_base64)
    return Image.open(BytesIO(image_data))


def explore_sample_structure(sample: dict, depth: int = 0, max_depth: int = 3) -> dict:
    """Recursively explore and return the structure of a sample."""
    if depth > max_depth:
        return "..."
    
    if isinstance(sample, dict):
        return {k: explore_sample_structure(v, depth + 1, max_depth) for k, v in sample.items()}
    elif isinstance(sample, list):
        if len(sample) == 0:
            return "[]"
        elif len(sample) == 1:
            return [explore_sample_structure(sample[0], depth + 1, max_depth)]
        else:
            return [explore_sample_structure(sample[0], depth + 1, max_depth), f"... ({len(sample)} items)"]
    elif isinstance(sample, str):
        if len(sample) > 100:
            return f"str[{len(sample)}]"
        return sample
    else:
        return type(sample).__name__


def verify_bbox_access(sample: dict) -> dict:
    """
    Verify that we can access bbox from pos_candidates.
    Returns info about the bbox structure found.
    """
    result = {
        "has_screenshot": False,
        "has_actions": False,
        "has_pos_candidates": False,
        "bbox_found": False,
        "bbox_format": None,
        "sample_bbox": None,
        "error": None
    }
    
    try:
        # Check screenshot
        if "screenshot" in sample:
            result["has_screenshot"] = True
            
        # Check actions structure
        if "actions" in sample and len(sample["actions"]) > 0:
            result["has_actions"] = True
            action = sample["actions"][0]
            
            # Look for pos_candidates in different possible locations
            pos_candidates = None
            
            # Try direct access
            if "pos_candidates" in action:
                pos_candidates = action["pos_candidates"]
            elif "pos_candidates" in sample:
                pos_candidates = sample["pos_candidates"]
                
            if pos_candidates and len(pos_candidates) > 0:
                result["has_pos_candidates"] = True
                candidate = pos_candidates[0]
                
                # Look for bbox in candidate
                if isinstance(candidate, dict):
                    if "bbox" in candidate:
                        result["bbox_found"] = True
                        result["bbox_format"] = type(candidate["bbox"]).__name__
                        result["sample_bbox"] = candidate["bbox"]
                    elif "attributes" in candidate and isinstance(candidate["attributes"], dict):
                        if "bbox" in candidate["attributes"]:
                            result["bbox_found"] = True
                            result["bbox_format"] = f"attributes.bbox: {type(candidate['attributes']['bbox']).__name__}"
                            result["sample_bbox"] = candidate["attributes"]["bbox"]
                    
                    # Also check for bounding_box or other variants
                    for key in ["bounding_box", "rect", "position", "bounds"]:
                        if key in candidate:
                            result["bbox_found"] = True
                            result["bbox_format"] = f"{key}: {type(candidate[key]).__name__}"
                            result["sample_bbox"] = candidate[key]
                            break
                            
    except Exception as e:
        result["error"] = str(e)
    
    return result


def download_and_verify(num_samples: int = 5, save_samples: bool = True):
    """
    Download dataset and verify structure.
    
    Args:
        num_samples: Number of samples to examine
        save_samples: Whether to save sample data to disk
    """
    print("=" * 60)
    print("Downloading Multimodal-Mind2Web dataset...")
    print("=" * 60)
    
    # Load dataset with streaming
    try:
        dataset = load_dataset(
            "osunlp/Multimodal-Mind2Web",
            streaming=True
        )
        print("✓ Dataset loaded successfully (streaming mode)")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("\nTrying alternative: loading just train split...")
        try:
            dataset = load_dataset(
                "osunlp/Multimodal-Mind2Web",
                split="train",
                streaming=True
            )
            print("✓ Train split loaded successfully")
            # Wrap in dict for consistent handling
            dataset = {"train": dataset}
        except Exception as e2:
            print(f"✗ Error loading train split: {e2}")
            return None
    
    print(f"\nAvailable splits: {list(dataset.keys()) if hasattr(dataset, 'keys') else 'streaming'}")
    
    # Get samples from train split
    print(f"\n{'=' * 60}")
    print(f"Examining {num_samples} samples from train split...")
    print("=" * 60)
    
    train_data = dataset["train"] if hasattr(dataset, '__getitem__') else dataset
    
    samples = []
    bbox_results = []
    
    for i, sample in enumerate(train_data):
        if i >= num_samples:
            break
            
        samples.append(sample)
        
        print(f"\n--- Sample {i + 1} ---")
        
        # Basic info
        if "annotation_id" in sample:
            print(f"Annotation ID: {sample['annotation_id']}")
        if "website" in sample:
            print(f"Website: {sample['website']}")
        if "confirmed_task" in sample:
            task = sample['confirmed_task']
            print(f"Task: {task[:100]}..." if len(task) > 100 else f"Task: {task}")
        
        # Verify bbox access
        bbox_result = verify_bbox_access(sample)
        bbox_results.append(bbox_result)
        
        print(f"\nBBox Verification:")
        print(f"  - Has screenshot: {bbox_result['has_screenshot']}")
        print(f"  - Has actions: {bbox_result['has_actions']}")
        print(f"  - Has pos_candidates: {bbox_result['has_pos_candidates']}")
        print(f"  - BBox found: {bbox_result['bbox_found']}")
        if bbox_result['bbox_found']:
            print(f"  - BBox format: {bbox_result['bbox_format']}")
            print(f"  - Sample bbox: {bbox_result['sample_bbox']}")
        if bbox_result['error']:
            print(f"  - Error: {bbox_result['error']}")
    
    # Explore full structure of first sample
    print(f"\n{'=' * 60}")
    print("Full structure of first sample:")
    print("=" * 60)
    if samples:
        structure = explore_sample_structure(samples[0])
        print(json.dumps(structure, indent=2, default=str))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    
    bbox_found_count = sum(1 for r in bbox_results if r['bbox_found'])
    print(f"Samples with bbox found: {bbox_found_count}/{len(bbox_results)}")
    
    if bbox_found_count == 0:
        print("\n⚠️  WARNING: No bbox found in samples!")
        print("The dataset structure may be different than expected.")
        print("You may need to:")
        print("  1. Check the raw HTML for element coordinates")
        print("  2. Use a separate candidate file from Mind2Web GitHub")
        print("  3. Compute bboxes using a headless browser")
    else:
        print(f"\n✓ BBox access verified successfully!")
        print(f"  Format: {bbox_results[0]['bbox_format']}")
    
    # Save samples for offline testing
    if save_samples and samples:
        output_dir = Path(__file__).parent / "samples"
        output_dir.mkdir(exist_ok=True)
        
        for i, sample in enumerate(samples):
            # Save metadata (excluding large fields)
            metadata = {k: v for k, v in sample.items() 
                       if k not in ['screenshot', 'raw_html', 'cleaned_html']}
            
            with open(output_dir / f"sample_{i}_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Save screenshot if available
            if "screenshot" in sample and sample["screenshot"]:
                try:
                    img = decode_screenshot(sample["screenshot"])
                    img.save(output_dir / f"sample_{i}_screenshot.png")
                    print(f"✓ Saved sample {i} screenshot ({img.size})")
                except Exception as e:
                    print(f"✗ Error saving screenshot {i}: {e}")
        
        print(f"\n✓ Samples saved to {output_dir}")
    
    return {
        "samples": samples,
        "bbox_results": bbox_results,
        "structure": structure if samples else None
    }


def quick_structure_test(num_samples: int = 50, save_samples: bool = True):
    """
    Test dataset structure with a reasonable sample size for local verification.
    Downloads samples and saves them locally for offline testing.
    
    Args:
        num_samples: Number of samples to fetch (default 50)
        save_samples: Whether to save samples to disk for offline use
    """
    print("=" * 60)
    print(f"DATASET STRUCTURE TEST ({num_samples} samples)")
    print("=" * 60)
    
    from datasets import load_dataset
    from pathlib import Path
    from tqdm import tqdm
    import json
    
    # Load with streaming
    print("Connecting to HuggingFace...")
    dataset = load_dataset(
        "osunlp/Multimodal-Mind2Web",
        split="train",
        streaming=True
    )
    print("✓ Dataset connection successful\n")
    
    samples = []
    bbox_found_count = 0
    screenshot_found_count = 0
    
    print(f"Fetching {num_samples} samples...")
    
    for i, sample in enumerate(tqdm(dataset, total=num_samples, desc="Downloading")):
        if i >= num_samples:
            break
        
        samples.append(sample)
        
        # Track stats
        if "screenshot" in sample and sample["screenshot"]:
            screenshot_found_count += 1
        
        # Check for bbox in pos_candidates (at top level)
        if "pos_candidates" in sample and sample["pos_candidates"]:
            cand = sample["pos_candidates"][0]
            if isinstance(cand, dict):
                # Check various possible bbox field names
                bbox_keys = ['bbox', 'bounding_box', 'rect', 'bounds', 'box']
                for bkey in bbox_keys:
                    if bkey in cand:
                        bbox_found_count += 1
                        break
                # Also check if there's coordinate-like data in attributes
                if 'attributes' in cand and isinstance(cand['attributes'], dict):
                    attrs = cand['attributes']
                    if any(k in attrs for k in bbox_keys):
                        bbox_found_count += 1
    
    print(f"\n✓ Downloaded {len(samples)} samples")
    
    # Show structure of first sample
    print(f"\n{'=' * 60}")
    print("SAMPLE STRUCTURE (first sample)")
    print("=" * 60)
    
    if samples:
        sample = samples[0]
        print(f"Top-level keys: {list(sample.keys())}")
        
        if "confirmed_task" in sample:
            print(f"\nTask: {sample['confirmed_task']}")
        
        if "website" in sample:
            print(f"Website: {sample['website']}")
        
        if "screenshot" in sample:
            screenshot = sample["screenshot"]
            if isinstance(screenshot, bytes):
                print(f"Screenshot: bytes (len={len(screenshot)})")
            elif isinstance(screenshot, str):
                print(f"Screenshot: base64 string (len={len(screenshot)})")
            else:
                print(f"Screenshot: {type(screenshot).__name__}")
        
        if "actions" in sample and sample["actions"]:
            print(f"\nActions: {len(sample['actions'])} steps")
            action = sample["actions"][0]
            print(f"  Action keys: {list(action.keys())}")
            
            if "operation" in action:
                print(f"  Operation: {action['operation']}")
            
        # Check candidates at TOP LEVEL (not nested in actions)
        for key in ['pos_candidates', 'neg_candidates']:
            if key in sample and sample[key]:
                candidates = sample[key]
                print(f"\n{key}: {len(candidates)} items")
                cand = candidates[0]
                print(f"  Type: {type(cand)}")
                if isinstance(cand, dict):
                    print(f"  Candidate keys: {list(cand.keys())}")
                    # Print all fields to understand structure
                    for k, v in cand.items():
                        if isinstance(v, str) and len(v) > 100:
                            print(f"    {k}: <string, len={len(v)}>")
                        elif isinstance(v, (list, dict)):
                            print(f"    {k}: {type(v).__name__} = {str(v)[:100]}...")
                        else:
                            print(f"    {k}: {v}")
                elif isinstance(cand, (list, tuple)):
                    print(f"  First candidate (list/tuple): {cand}")
                else:
                    print(f"  First candidate: {cand}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")
    print(f"With screenshots: {screenshot_found_count}/{len(samples)}")
    print(f"With bbox in pos_candidates: {bbox_found_count}/{len(samples)}")
    
    if bbox_found_count > 0:
        print("\n✓ BBOX ACCESS CONFIRMED - Ready for VGAP training!")
    else:
        print("\n⚠️  WARNING: No bbox found - need to investigate structure")
    
    # Save samples for offline testing
    if save_samples and samples:
        output_dir = Path(__file__).parent / "samples"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving samples to {output_dir}...")
        
        # Save metadata for all samples (excluding large fields)
        metadata_list = []
        for i, sample in enumerate(samples):
            metadata = {}
            for k, v in sample.items():
                if k in ['screenshot']:
                    metadata[k] = f"<{type(v).__name__}, len={len(str(v))}>"
                elif k in ['raw_html', 'cleaned_html']:
                    metadata[k] = f"<string, len={len(str(v))}>" if v else None
                else:
                    metadata[k] = v
            metadata_list.append(metadata)
        
        with open(output_dir / "samples_metadata.json", "w") as f:
            json.dump(metadata_list, f, indent=2, default=str)
        
        # Save ALL screenshots
        print(f"Saving {len(samples)} screenshots...")
        saved_count = 0
        for i, sample in enumerate(tqdm(samples, desc="Saving screenshots")):
            if "screenshot" in sample and sample["screenshot"]:
                try:
                    screenshot = sample["screenshot"]
                    # Check if it's already a PIL Image or needs decoding
                    if hasattr(screenshot, 'save'):
                        # Already a PIL Image
                        screenshot.save(output_dir / f"sample_{i}_screenshot.png")
                        saved_count += 1
                    elif isinstance(screenshot, (str, bytes)):
                        # Base64 or bytes - decode first
                        img = decode_screenshot(screenshot)
                        img.save(output_dir / f"sample_{i}_screenshot.png")
                        saved_count += 1
                    else:
                        print(f"  Unknown screenshot type {i}: {type(screenshot)}")
                except Exception as e:
                    print(f"  Could not save screenshot {i}: {e}")
        print(f"✓ Saved metadata and {saved_count} screenshots to {output_dir}")
    
    return samples


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Full verification (downloads more data, detailed analysis)
        result = download_and_verify(num_samples=10, save_samples=True)
    else:
        # Default: fetch 50 samples for local testing
        num = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 50
        samples = quick_structure_test(num_samples=num, save_samples=True)

