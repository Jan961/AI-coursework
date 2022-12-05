import numpy as np

from GridState import GridState
from Grid import Grid


def split(grid : Grid, random = False):
    for idx in np.flip(np.argsort(grid.rectangle_list[:,2])):
        poss_splits = []
        h, w = grid.rectangle_list[idx][0], grid.rectangle_list[idx][1]
        if h> 2:
            for i in range(1,h):
                state = grid.get_potential_grid_state(idx, i, False, split=True)  #false means: vertical=False too late
                if state.is_valid :                                          # to change to a kw arg for legibility
                    poss_splits.append((idx, i, False, state.Mond_score))
        if w>2 :
            for i in range(1,w):
                state = grid.get_potential_grid_state(idx, i, True, split=True)  #true means: vertical=True
                if state.is_valid :
                    poss_splits.append((idx, i, True, state.Mond_score))

        if len(poss_splits) > 0:
            if not random:
                max = sorted(poss_splits, key=lambda x:x[3])[0]
                grid.cleave_rectangle(*max[:3])
            else:
                rand_choice = poss_splits[np.random.randint(0, len(poss_splits))]
                grid.cleave_rectangle(*rand_choice[:3])

            return True

        else:
            return False




def merge(grid: Grid, random=False) :
    poss_merges = []
    vertical, horizontal = grid.get_lists_to_merge()
    print(f"verct: {vertical}, hor: {horizontal}")

    for ids in vertical:
        print(ids)
        state = grid.get_potential_grid_state(ids[0], ids[1], True, split=False) #True means: vertical = True
        print(state)
        if state.is_valid:
            poss_merges.append((ids[0], ids[1], True, state.Mond_score))
    for ids in horizontal:
        state = grid.get_potential_grid_state(ids[0], ids[1], False, split=False)
        print(state)
        print(ids)
        if state.is_valid:
            poss_merges.append((ids[0], ids[1], False, state.Mond_score))

    if len(poss_merges) > 0:

        if not random:
            max = sorted(poss_merges, key=lambda x: x[3])[0]
            grid.merge_rectangles(*max[:3])
        else:
            rand_choice = poss_merges[np.random.randint(0, len(poss_merges))]
            grid.merge_rectangles(*rand_choice[:3])

        return True

    else:
        return False
