"""
Create DPO preference pairs for VGAP training.

Extracts bounding boxes from pos_candidates and creates:
- Preferred: tight crop around target element
- Rejected: random/full/oversized crops
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class BoundingBox:
    """Bounding box with x1, y1, x2, y2 coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> 'BoundingBox':
        """Create from x, y, width, height format."""
        return cls(x1=x, y1=y, x2=x + w, y2=y + h)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_crop_string(self) -> str:
        """Format as CROP(x1,y1,x2,y2)."""
        return f"CROP({int(self.x1)},{int(self.y1)},{int(self.x2)},{int(self.y2)})"
    
    def contains(self, other: 'BoundingBox') -> bool:
        """Check if this bbox fully contains another."""
        return (self.x1 <= other.x1 and self.y1 <= other.y1 and 
                self.x2 >= other.x2 and self.y2 >= other.y2)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Compute intersection over union."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        
        if ix2 < ix1 or iy2 < iy1:
            return 0.0
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0
    
    def expand(self, padding_x: float, padding_y: float = None) -> 'BoundingBox':
        """Expand bbox by padding on all sides."""
        if padding_y is None:
            padding_y = padding_x
        return BoundingBox(
            x1=self.x1 - padding_x,
            y1=self.y1 - padding_y,
            x2=self.x2 + padding_x,
            y2=self.y2 + padding_y
        )
    
    def clamp(self, max_w: float, max_h: float) -> 'BoundingBox':
        """Clamp bbox to image bounds."""
        return BoundingBox(
            x1=max(0, self.x1),
            y1=max(0, self.y1),
            x2=min(max_w, self.x2),
            y2=min(max_h, self.y2)
        )


def parse_candidate(candidate_str: str) -> Optional[Dict]:
    """Parse a candidate JSON string and extract bbox."""
    try:
        cand = json.loads(candidate_str)
        attrs_str = cand.get('attributes', '{}')
        attrs = json.loads(attrs_str) if isinstance(attrs_str, str) else attrs_str
        
        bbox_str = attrs.get('bounding_box_rect')
        if bbox_str:
            # Format: "x,y,width,height"
            parts = [float(p) for p in bbox_str.split(',')]
            if len(parts) == 4:
                x, y, w, h = parts
                return {
                    'tag': cand.get('tag', ''),
                    'bbox': BoundingBox.from_xywh(x, y, w, h),
                    'backend_node_id': cand.get('backend_node_id', ''),
                    'attributes': attrs,
                    'is_target': cand.get('is_top_level_target', False)
                }
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        pass
    return None


def create_input_prompt(task: str, candidates: List[Dict], screen_size: Tuple[int, int]) -> str:
    """Create the input prompt for VGAP model."""
    width, height = screen_size
    
    lines = [f"Task: {task}", f"Screen: {width}x{height}", "Candidates:"]
    
    for i, cand in enumerate(candidates[:20]):  # Limit to top 20 candidates
        tag = cand['tag']
        bbox = cand['bbox']
        attrs = cand['attributes']
        
        # Extract useful attributes
        elem_id = attrs.get('id', '')
        elem_class = attrs.get('class', '')[:50]  # Truncate long classes
        aria_label = attrs.get('aria_label', '')[:50]
        
        attr_str = []
        if elem_id:
            attr_str.append(f'id="{elem_id}"')
        if elem_class:
            attr_str.append(f'class="{elem_class}"')
        if aria_label:
            attr_str.append(f'aria="{aria_label}"')
        
        lines.append(f"[{i+1}] <{tag} {' '.join(attr_str)}> bbox:({int(bbox.x1)},{int(bbox.y1)},{int(bbox.x2)},{int(bbox.y2)})")
    
    return "\n".join(lines)


def get_screen_size_from_candidates(candidates: List[str]) -> Tuple[int, int]:
    """Extract actual screen size from candidate bboxes (find the largest one)."""
    max_width = 1280
    max_height = 720
    
    for cand_str in candidates[:100]:  # Check first 100 candidates
        try:
            cand = json.loads(cand_str)
            attrs_str = cand.get('attributes', '{}')
            attrs = json.loads(attrs_str) if isinstance(attrs_str, str) else attrs_str
            
            bbox_str = attrs.get('bounding_box_rect')
            if bbox_str:
                parts = [float(p) for p in bbox_str.split(',')]
                if len(parts) == 4:
                    x, y, w, h = parts
                    max_width = max(max_width, int(x + w))
                    max_height = max(max_height, int(y + h))
        except:
            pass
    
    return (max_width, max_height)


def generate_rejected_crops(
    target_bbox: BoundingBox,
    screen_size: Tuple[int, int],
    num_rejected: int = 3
) -> List[Tuple[str, BoundingBox]]:
    """Generate rejected crop options."""
    width, height = screen_size
    rejected = []
    
    # 1. Full screenshot (inefficient) - use ACTUAL page dimensions
    full_bbox = BoundingBox(0, 0, width, height)
    rejected.append(("full_page", full_bbox))
    
    # 2. Random region not containing target
    for _ in range(num_rejected - 2):
        # Generate random bbox that doesn't contain target
        for attempt in range(10):
            rw = random.randint(100, width // 2)
            rh = random.randint(100, height // 2)
            rx = random.randint(0, width - rw)
            ry = random.randint(0, height - rh)
            random_bbox = BoundingBox(rx, ry, rx + rw, ry + rh)
            
            # Make sure it doesn't contain the target
            if random_bbox.iou(target_bbox) < 0.3:
                rejected.append(("random", random_bbox))
                break
    
    # 3. Overly large crop (wasteful)
    padding = max(target_bbox.width, target_bbox.height) * 2
    oversized = target_bbox.expand(padding).clamp(width, height)
    rejected.append(("oversized", oversized))
    
    return rejected


def create_preference_pair(
    sample: Dict,
    screen_size: Tuple[int, int] = None  # Auto-detect if None
) -> Optional[Dict]:
    """Create a single DPO preference pair from a sample."""

    # Parse positive candidates
    pos_candidates = sample.get('pos_candidates', [])
    if not pos_candidates:
        return None

    parsed_pos = []
    for cand_str in pos_candidates:
        parsed = parse_candidate(cand_str)
        if parsed:
            parsed_pos.append(parsed)

    if not parsed_pos:
        return None

    # Get target element (first positive candidate)
    target = parsed_pos[0]
    target_bbox = target['bbox']

    # Parse negative candidates for context
    neg_candidates = sample.get('neg_candidates', [])
    parsed_neg = []
    for cand_str in neg_candidates[:50]:  # Limit for efficiency
        parsed = parse_candidate(cand_str)
        if parsed:
            parsed_neg.append(parsed)

    # Auto-detect screen size from candidates (full page dimensions)
    if screen_size is None:
        all_candidates = pos_candidates + neg_candidates
        screen_size = get_screen_size_from_candidates(all_candidates)

    # Create input prompt
    all_candidates = parsed_pos + parsed_neg
    task = sample.get('confirmed_task', '')
    input_prompt = create_input_prompt(task, all_candidates, screen_size)

    # Preferred output: crop with REASONABLE context around target
    # Based on literature: VLMs need context, not just the element
    target_width = target_bbox.width
    target_height = target_bbox.height

    # Context padding: aim for 3-5x the target element size for meaningful context
    context_factor = 3.0  # 3x the target element dimensions
    padding_x = max(target_width * (context_factor - 1) / 2, 100)  # minimum 100px
    padding_y = max(target_height * (context_factor - 1) / 2, 100)  # minimum 100px

    preferred_bbox = target_bbox.expand(padding_x, padding_y).clamp(screen_size[0], screen_size[1])
    preferred_output = preferred_bbox.to_crop_string()

    # Generate rejected outputs
    rejected_crops = generate_rejected_crops(target_bbox, screen_size)

    # Create multiple preference pairs (one per rejected option)
    pairs = []
    for reject_type, rejected_bbox in rejected_crops:
        pairs.append({
            'prompt': input_prompt,
            'chosen': preferred_output,
            'rejected': rejected_bbox.to_crop_string(),
            'reject_type': reject_type,
            'target_bbox': {
                'x1': target_bbox.x1, 'y1': target_bbox.y1,
                'x2': target_bbox.x2, 'y2': target_bbox.y2
            },
            'preferred_context': {
                'width': preferred_bbox.width,
                'height': preferred_bbox.height,
                'area': preferred_bbox.area,
                'context_factor': context_factor
            },
            'metadata': {
                'task': task,
                'website': sample.get('website', ''),
                'annotation_id': sample.get('annotation_id', ''),
                'target_tag': target['tag']
            }
        })

    return pairs


def process_dataset(
    num_samples: int = None,
    output_path: str = "data/dpo_preferences.json",
    split: str = "train"
):
    """Process dataset and create DPO preference pairs."""
    
    print("=" * 60)
    print("Creating DPO Preference Pairs")
    print("=" * 60)
    print(f"  Dataset: osunlp/Multimodal-Mind2Web")
    print(f"  Split: {split}")
    print(f"  Samples: {num_samples if num_samples else 'all'}")
    print(f"  Output: {output_path}")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading {split} split from HuggingFace...")
    dataset = load_dataset(
        "osunlp/Multimodal-Mind2Web",
        split=split,
        streaming=True
    )
    
    all_pairs = []
    success_count = 0
    fail_count = 0
    
    desc = f"Processing {num_samples if num_samples else 'all'} samples"
    
    for i, sample in enumerate(tqdm(dataset, total=num_samples, desc=desc)):
        if num_samples and i >= num_samples:
            break
        
        # Let create_preference_pair auto-detect screen size from bboxes
        pairs = create_preference_pair(sample)  # Auto-detects full page dimensions
        if pairs:
            all_pairs.extend(pairs)
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n✓ Created {len(all_pairs)} preference pairs")
    print(f"  Successful samples: {success_count}")
    print(f"  Failed samples: {fail_count}")
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_pairs, f, indent=2)
    
    print(f"✓ Saved to {output_path}")
    
    # Show sample
    if all_pairs:
        print(f"\n{'=' * 60}")
        print("Sample preference pair:")
        print("=" * 60)
        sample_pair = all_pairs[0]
        print(f"Prompt (first 500 chars):\n{sample_pair['prompt'][:500]}...")
        print(f"\nChosen: {sample_pair['chosen']}")
        print(f"Rejected: {sample_pair['rejected']} (type: {sample_pair['reject_type']})")
    
    return all_pairs


def process_local_samples(samples_path: str = "data/samples/samples_metadata.json"):
    """Process locally saved samples (for testing without network)."""
    
    print("=" * 60)
    print("Processing Local Samples")
    print("=" * 60)
    
    with open(samples_path) as f:
        samples = json.load(f)
    
    print(f"Loaded {len(samples)} samples from {samples_path}")
    
    all_pairs = []
    success_count = 0
    
    for sample in tqdm(samples, desc="Processing"):
        pairs = create_preference_pair(sample)  # Auto-detects full page dimensions
        if pairs:
            all_pairs.extend(pairs)
            success_count += 1
    
    print(f"\n✓ Created {len(all_pairs)} preference pairs from {success_count} samples")
    
    # Save
    output_path = Path("data/dpo_preferences_local.json")
    with open(output_path, 'w') as f:
        json.dump(all_pairs, f, indent=2)
    
    print(f"✓ Saved to {output_path}")
    
    return all_pairs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create DPO preference pairs from Multimodal-Mind2Web dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 2000 training examples
  python create_preferences.py --split train --num_samples 2000 --output data/dpo_train_2k.json
  
  # Generate 200 test examples
  python create_preferences.py --split test_task --num_samples 200 --output data/dpo_test_200.json
  
  # Generate all training data
  python create_preferences.py --split train --output data/dpo_train_full.json
  
  # Process local samples (offline)
  python create_preferences.py --local
        """
    )
    
    parser.add_argument(
        "--split", 
        type=str, 
        default="train",
        choices=["train", "test_task", "test_website", "test_domain"],
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=None,
        help="Number of samples to process (default: all)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output JSON file path (default: data/dpo_preferences_{split}.json)"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Process local samples only (from data/samples/samples_metadata.json)"
    )
    
    args = parser.parse_args()
    
    if args.local:
        # Process local samples only
        process_local_samples()
    else:
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            suffix = f"_{args.num_samples}" if args.num_samples else "_full"
            output_path = f"data/dpo_preferences_{args.split}{suffix}.json"
        
        # Process from HuggingFace
        process_dataset(
            num_samples=args.num_samples,
            output_path=output_path,
            split=args.split
        )

