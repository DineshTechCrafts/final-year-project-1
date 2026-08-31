"""
Tests for deterministic patient splitting logic.
"""
import json
import random
import pytest

def test_deterministic_split():
    """Verify that patient splitting is deterministic and has no leakage."""
    # Simulate patients
    patients = [f"p_{i:03d}" for i in range(100)]
    
    def perform_split(p_list):
        # Must sort to guarantee determinism regardless of input set order
        p_list = sorted(list(p_list))
        random.seed(42)
        random.shuffle(p_list)
        
        n = len(p_list)
        n_tr = int(n * 0.70)
        n_val = int(n * 0.15)
        
        return {
            "train": set(p_list[:n_tr]),
            "val": set(p_list[n_tr:n_tr+n_val]),
            "test": set(p_list[n_tr+n_val:])
        }
        
    split1 = perform_split(patients)
    split2 = perform_split(patients)
    
    # Check determinism
    assert split1["train"] == split2["train"]
    assert split1["val"] == split2["val"]
    assert split1["test"] == split2["test"]
    
    # Check no leakage (intersection is empty)
    assert len(split1["train"].intersection(split1["val"])) == 0
    assert len(split1["train"].intersection(split1["test"])) == 0
    assert len(split1["val"].intersection(split1["test"])) == 0
    
    # Check all patients accounted for
    assert len(split1["train"]) + len(split1["val"]) + len(split1["test"]) == 100
