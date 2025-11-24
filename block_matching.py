import numpy as np

def SAD(block1, block2):
    return np.sum(np.abs(block1.astype(int) - block2.astype(int)))

def search_block(prev_frame, curr_frame, x, y, block_size, search_range):
    h, w = curr_frame.shape
    best_sad = 1e9
    best_dx, best_dy = 0, 0
    sp_count = 0 

    block = curr_frame[y:y+block_size, x:x+block_size]

    for dy in range(-search_range, search_range+1):
        for dx in range(-search_range, search_range+1):
            
            sp_count += 1

            ny, nx = y + dy, x + dx

            if nx < 0 or ny < 0 or nx+block_size >= w or ny+block_size >= h:
                continue

            cand = prev_frame[ny:ny+block_size, nx:nx+block_size]
            sad = SAD(block, cand)

            if sad < best_sad:
                best_sad = sad
                best_dx, best_dy = dx, dy

            if sad < 50:  # early termination
                return best_dx, best_dy , sp_count

    return best_dx, best_dy , sp_count 
