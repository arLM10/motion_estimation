# classical_methods.py

import numpy as np
# Assuming SAD is imported from block_matching or defined locally
from block_matching import SAD

# --- Three-Step Search (TSS) - Modified to return SP Count ---
def three_step_search(prev, curr, x, y, block_size, step=4):
    best_dx, best_dy = 0, 0
    best_sad = 1e9
    sp_count = 0
    block = curr[y:y+block_size, x:x+block_size]

    for s in [step, step//2, step//4]:
        # TSS checks 9 points in each of the 3 steps (total 25 max, excluding center repetition)
        for dx in [-s, 0, s]:
            for dy in [-s, 0, s]:
                if s != step or (dx != 0 or dy != 0): # Avoid re-checking center point in steps 2 and 3
                    sp_count += 1
                
                nx = x + best_dx + dx
                ny = y + best_dy + dy
                
                if nx < 0 or ny < 0 or nx+block_size >= curr.shape[1] or ny+block_size >= curr.shape[0]:
                    continue
                
                cand = prev[ny:ny+block_size, nx:nx+block_size]
                sad = SAD(block, cand)

                if sad < best_sad:
                    best_sad = sad
                    best_dx_temp = dx
                    best_dy_temp = dy
        
        # Update center for the next step only if a better point was found
        if 'best_dx_temp' in locals():
             best_dx += best_dx_temp
             best_dy += best_dy_temp
             del best_dx_temp, best_dy_temp


    return best_dx, best_dy, sp_count


# --- Diamond Search (DS) - Implementation ---
def diamond_search(prev, curr, x, y, block_size, search_range=16):
    best_dx, best_dy = 0, 0
    best_sad = 1e9
    sp_count = 0
    block = curr[y:y+block_size, x:x+block_size]
    
    # Large Diamond Search Pattern (LDP) points relative to the center (0,0)
    LDP = [(0, 0), (0, 2), (0, -2), (2, 0), (-2, 0)]
    # Small Diamond Search Pattern (SDP) points relative to the center (0,0)
    SDP = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # 1. Initial Search (Large Diamond)
    search_center_dx, search_center_dy = 0, 0
    
    # Helper to check a pattern and update best MV and SAD
    def check_pattern(center_dx, center_dy, pattern, prev, curr, x, y, block_size, best_sad, best_dx, best_dy, sp_count):
        found_new_best = False
        
        for dx, dy in pattern:
            sp_count += 1
            nx = x + center_dx + dx
            ny = y + center_dy + dy
            
            # Boundary Check
            if nx < 0 or ny < 0 or nx+block_size >= curr.shape[1] or ny+block_size >= curr.shape[0]:
                continue
                
            cand = prev[ny:ny+block_size, nx:nx+block_size]
            sad = SAD(block, cand)

            if sad < best_sad:
                best_sad = sad
                best_dx = center_dx + dx
                best_dy = center_dy + dy
                found_new_best = True
        
        return best_sad, best_dx, best_dy, sp_count, found_new_best

    # Initial check (The center (0,0) is included in LDP, so LDP is checked once)
    best_sad, best_dx, best_dy, sp_count, found_new_best = check_pattern(
        search_center_dx, search_center_dy, LDP, prev, curr, x, y, block_size, best_sad, best_dx, best_dy, sp_count
    )
    
    # 2. Iterative Large Diamond Search
    # The process repeats until the best match is the center of the pattern
    while found_new_best and abs(best_dx) <= search_range and abs(best_dy) <= search_range:
        search_center_dx, search_center_dy = best_dx, best_dy
        found_new_best = False
        
        # Check the four new LDP points around the new center
        best_sad, best_dx_temp, best_dy_temp, sp_count, found_new_best_LDP = check_pattern(
            search_center_dx, search_center_dy, 
            [(0, 2), (0, -2), (2, 0), (-2, 0)], # Check 4 points only (center and previous checked points are excluded)
            prev, curr, x, y, block_size, best_sad, best_dx, best_dy, sp_count
        )
        # Update MVs only if a better point was found
        if found_new_best_LDP:
            best_dx, best_dy = best_dx_temp, best_dy_temp
            found_new_best = True
            
    # 3. Final Search (Small Diamond)
    # The search is confined to the 5 points of the Small Diamond around the current best match
    best_sad, best_dx, best_dy, sp_count, _ = check_pattern(
        best_dx, best_dy, SDP, prev, curr, x, y, block_size, best_sad, best_dx, best_dy, sp_count
    )

    return best_dx, best_dy, sp_count

# --- Note: You would similarly implement hexagon_search (HS) and test_zone_search (TZS) here ---
# (Implementations omitted for brevity, but you must add them to complete the comparison)
# def hexagon_search(...):
#     # ... implementation ...
#     return best_dx, best_dy, sp_count
#
# def test_zone_search(...):
#     # ... implementation ...
#     return best_dx, best_dy, sp_count