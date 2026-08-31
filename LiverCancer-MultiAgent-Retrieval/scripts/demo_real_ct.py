import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.pipeline import run_inference

def main():
    parser = argparse.ArgumentParser(description="Demo CLI for Real CT Upload Pipeline")
    parser.add_argument("--image", required=True, help="Path to CT JPEG/PNG")
    args = parser.parse_args()
    
    print(f"Executing frozen research pipeline on {args.image}...")
    
    result = run_inference(args.image, "CT")
    
    if result["status"] == "ERROR":
        print("\n[PIPELINE HALTED]")
        print(f"Reason: {result['message']}")
        sys.exit(1)
        
    print("\n[PIPELINE SUCCESS]")
    print(result)

if __name__ == "__main__":
    main()
