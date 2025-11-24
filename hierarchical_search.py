# hierarchical_search.py

# Note: The import assumes search_block now returns (dx, dy, sp_count)
from block_matching import search_block 

def hierarchical_ME(prev_pyr, curr_pyr, x, y, block_size, search_range):
    total_sp = 0
    
    # Level 2
    dx2, dy2, sp2 = search_block(prev_pyr[2], curr_pyr[2], x//4, y//4, block_size//4, search_range//4)
    total_sp += sp2

    # Level 1
    dx1, dy1, sp1 = search_block(prev_pyr[1], curr_pyr[1],
                            x//2 + 2*dx2, y//2 + 2*dy2,
                            block_size//2, search_range//2)
    total_sp += sp1

    # Level 0
    dx0, dy0, sp0 = search_block(prev_pyr[0], curr_pyr[0],
                            x + dx1*2, y + dy1*2,
                            block_size, search_range)
    total_sp += sp0

    # Return the Motion Vector and the total Search Points count
    return dx0, dy0, total_sp