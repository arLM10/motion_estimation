# main.py

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
# Import all necessary functions
from video_loader import load_video_frames
from utils import build_pyramid
from texture_analysis import compute_texture
# Note: hierarchical_ME now returns 3 values
from hierarchical_search import hierarchical_ME 
from evaluation_metrics import psnr
from classical_methods import three_step_search, diamond_search # Import TZS and HS when implemented!


VIDEO_PATH = "/home/fdbdfg/Desktop/motion/akiyo_qcif.y4m"   # change this
BLOCK_SIZE = 16
SEARCH_RANGE_MAX = 16 # Used for classical methods

# --- Helper Function: Motion Compensation ---
# Uses the MVs to reconstruct the current frame from the previous frame.
def motion_compensate(prev_frame, mv_frame, block_size):
    h, w = prev_frame.shape
    predicted_frame = np.zeros_like(prev_frame)
    
    # Iterate through the motion vectors grid
    for y_idx, row in enumerate(mv_frame):
        for x_idx, mv in enumerate(row):
            dx, dy = mv
            
            # Block coordinates in the current frame
            y = y_idx * block_size
            x = x_idx * block_size
            
            # Corresponding block coordinates in the previous frame
            ny = y + dy
            nx = x + dx
            
            # Boundary check for the candidate block in the previous frame
            if 0 <= ny and ny + block_size <= h and 0 <= nx and nx + block_size <= w:
                # Copy the block from the previous frame to the predicted frame
                predicted_frame[y:y+block_size, x:x+block_size] = prev_frame[ny:ny+block_size, nx:nx+block_size]
            else:
                # For blocks where the MV goes out of bounds, just copy the corresponding block (zero MV)
                predicted_frame[y:y+block_size, x:x+block_size] = prev_frame[y:y+block_size, x:x+block_size]

    return predicted_frame.astype(np.uint8)


# --- Function to run a single ME algorithm across the whole video ---
def run_me_algorithm(frames, algorithm_name, block_size):
    print(f"Running {algorithm_name}...")
    
    all_psnr = []
    total_sp = 0
    
    start_time = time.time()
    
    for i in range(1, len(frames)):
        prev = frames[i-1]
        curr = frames[i]
        
        h, w = curr.shape
        mv_frame = []
        
        # --- Algorithm Selection and Per-Block Execution ---
        if algorithm_name == "Hierarchical_Adaptive":
            prev_pyr = build_pyramid(prev)
            curr_pyr = build_pyramid(curr)
            
            current_frame_sp = 0
            
            for y in range(0, h, block_size):
                row = []
                for x in range(0, w, block_size):
                    block = curr[y:y+block_size, x:x+block_size]
                    texture = compute_texture(block)

                    if texture < 5:
                        search_range = 4
                    elif texture < 20:
                        search_range = 8
                    else:
                        search_range = 16

                    # Note the change: hierarchical_ME now returns 3 values
                    dx, dy, sp_count = hierarchical_ME(prev_pyr, curr_pyr, x, y, block_size, search_range)
                    row.append((dx, dy))
                    current_frame_sp += sp_count
                mv_frame.append(row)
            
            total_sp += current_frame_sp

        elif algorithm_name == "TSS":
            current_frame_sp = 0
            for y in range(0, h, block_size):
                row = []
                for x in range(0, w, block_size):
                    # Note the change: three_step_search now returns 3 values
                    dx, dy, sp_count = three_step_search(prev, curr, x, y, block_size, step=4)
                    row.append((dx, dy))
                    current_frame_sp += sp_count
                mv_frame.append(row)
            total_sp += current_frame_sp
            
        elif algorithm_name == "DS":
            current_frame_sp = 0
            for y in range(0, h, block_size):
                row = []
                for x in range(0, w, block_size):
                    # Note the change: diamond_search now returns 3 values
                    dx, dy, sp_count = diamond_search(prev, curr, x, y, block_size, SEARCH_RANGE_MAX)
                    row.append((dx, dy))
                    current_frame_sp += sp_count
                mv_frame.append(row)
            total_sp += current_frame_sp

        # Add "HS" and "TZS" logic here when those functions are implemented in classical_methods.py!
        # --- END OF ALGORITHM EXECUTION ---
        
        # --- PSNR Calculation ---
        predicted_frame = motion_compensate(prev, mv_frame, block_size)
        frame_psnr = psnr(curr, predicted_frame)
        all_psnr.append(frame_psnr)

    end_time = time.time()
    
    # --- Final Analysis ---
    num_blocks = (h // block_size) * (w // block_size) * (len(frames) - 1)
    
    avg_psnr = np.mean(all_psnr)
    runtime = end_time - start_time
    avg_sp_per_mb = total_sp / num_blocks if num_blocks > 0 else 0
    
    return avg_psnr, runtime, avg_sp_per_mb

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # Check if video file exists before proceeding (you must place your_video.mp4 here)
    try:
        frames = load_video_frames(VIDEO_PATH)
        if not frames:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"ERROR: Video file not found at {VIDEO_PATH}. Please set VIDEO_PATH and ensure the file exists.")
        exit()
        
    print(f"Video loaded with {len(frames)} frames.")
    
    # List of algorithms to test. Add "HS" and "TZS" here when implemented!
    algorithms = ["Hierarchical_Adaptive", "TSS", "DS"] 
    
    results = {}

    for algo in algorithms:
        avg_psnr, runtime, avg_sp_per_mb = run_me_algorithm(frames, algo, BLOCK_SIZE)
        results[algo] = {
            'PSNR': avg_psnr,
            'Runtime': runtime,
            'SP/MB': avg_sp_per_mb
        }
    
    # --- Deliverable 1: Comparison Table ---
    print("\n" + "="*50)
    print("ME Algorithm Performance Comparison")
    print("="*50)
    print(f"{'Algorithm':<25}{'Avg PSNR (dB)':<15}{'Runtime (s)':<15}{'Avg SP/MB':<15}")
    print("-" * 70)
    for algo, res in results.items():
        print(f"{algo:<25}{res['PSNR']:<15.3f}{res['Runtime']:<15.3f}{res['SP/MB']:<15.2f}")
    
    # --- Deliverable 2: PSNR vs. Runtime Plot ---
    
    algos = list(results.keys())
    psnrs = [results[algo]['PSNR'] for algo in algos]
    runtimes = [results[algo]['Runtime'] for algo in algos]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(runtimes, psnrs, s=150, alpha=0.7)
    
    for i, algo in enumerate(algos):
        # Annotate points with the algorithm name
        plt.annotate(algo, (runtimes[i] * 1.05, psnrs[i]), fontsize=10)
        
    plt.title('PSNR vs. Runtime Trade-off for Motion Estimation')
    plt.xlabel('Total Runtime (seconds)')
    plt.ylabel('Average PSNR (dB)')
    plt.grid(True)
    plt.show()
    
    print("\nAnalysis complete! The comparison table and PSNR-Runtime plot are generated.")
    # You will need to write a separate script for the Motion Vector Visualization.