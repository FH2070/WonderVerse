import json
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def sort_by_id(extrinsics):
    return sorted(extrinsics, key=lambda x: x["id"])


def analyze_extrinsics(extrinsics):
    
    positions = [frame["position"] for frame in extrinsics]
    rotations = [frame["rotation"] for frame in extrinsics]

    
    translational_changes = [
        np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1])) 
        for i in range(1, len(positions))
    ]

    
    rotational_changes = []
    for i in range(1, len(rotations)):
        R_prev = R.from_matrix(rotations[i-1])
        R_curr = R.from_matrix(rotations[i])
        delta_R = (R_curr * R_prev.inv()).magnitude()
        rotational_changes.append(delta_R)

    return translational_changes, rotational_changes


def main():
    
    file_path = "gaussian-splatting-main/output/cam_filter/cameras.json" 
    extrinsics = load_json(file_path)

    
    extrinsics_sorted = sort_by_id(extrinsics)

    
    translational_changes, rotational_changes = analyze_extrinsics(extrinsics_sorted)

    
    print(translational_changes)
    print( rotational_changes)

    
    translation_threshold = 1  
    rotation_threshold = 0.25     ）

    translation_jumps = [i for i, d in enumerate(translational_changes) if d > translation_threshold]
    rotation_jumps = [i for i, a in enumerate(rotational_changes) if a > rotation_threshold]
    
    print( translation_jumps)
    print( rotation_jumps)

if __name__ == "__main__":
    main()
