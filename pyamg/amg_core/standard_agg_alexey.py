import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def standard_aggregation_py(n_row, Ap, Aj, x, y, coord=None, modify={'phase-1a': 0.0, 'phase-2': False}):

    x[:n_row]      = 0
    next_aggregate = 1; # number of aggregates + 1
    aggs_dict      = np.zeros((n_row,))
    # Pass 1
    for i in range(n_row):
        if x[i]:
            continue # alreay marked

        row_start = Ap[i]
        row_end   = Ap[i+1]
        #Determine whether all neighbors of this node are free (not already aggregates)
        has_aggregated_neighbors = False
        has_neighbors = False

        neigh_count = row_end-row_start
        agged_neigh = 0
        for jj in range(row_start, row_end):
            j = Aj[jj]
            if i != j:
                has_neighbors = True
                if x[j]:
                    agged_neigh +=1

        if agged_neigh/neigh_count > modify['phase-1a']:
            has_aggregated_neighbors = True

        if not has_neighbors:  #isolated node, do not aggregate
            x[i] = -n_row;
        elif not has_aggregated_neighbors: #Make an aggregate out of this node and its neighbors
            x[i] = next_aggregate
            aggs_dict[next_aggregate] += 1
            y[next_aggregate-1] = i # y stores a list of the Cpts

            for jj in range(row_start, row_end):
                if x[Aj[jj]] == 0:
                    x[Aj[jj]] = next_aggregate;
                    aggs_dict[next_aggregate] += 1
            next_aggregate+=1;

    pass1_naggs = len(x[:n_row][np.where(x >0 )])

    # Pass 2: Add unaggregated nodes to any neighboring aggregate
    if not modify['phase-2']:
        for i in range(n_row):
            if x[i]:
                continue # alreay marked
            for jj in range(Ap[i],  Ap[i+1]):
                j  = Aj[jj]
                xj = x[j]
                if xj > 0:
                    x[i] = -xj;
                    break;
    else:
        mean = np.mean(aggs_dict)
        std  = np.std(aggs_dict)
        #for i in range(n_row-1,-1,-1):
        for i in range(n_row):
            if x[i]:
                continue # alreay marked

            neighbors = []
            for jj in range(Ap[i],  Ap[i+1]):
                j  = Aj[jj]
                xj = x[j]
                if xj > 0:
                    neighbors.append(xj)

            agg_ids     = []
            conn_to_agg = []
            agg_size    = []
            countss     = np.bincount(neighbors)
            while np.sum(countss):
                agg_ids.append(np.argmax(countss))
                conn_to_agg.append(countss[agg_ids[-1]])
                countss[agg_ids[-1]] = 0
                agg_size.append(aggs_dict[agg_ids[-1]])

            m = np.array(agg_size) - mean
            m[np.where(m > 0)] = 1
            m = np.abs(m)
            score = np.array(agg_size)*np.array(conn_to_agg)*m
            score[np.where(agg_size > mean+std/2)] = -1
            pick = agg_ids[np.argmax(score)]
            x[i] = -pick

    next_aggregate-= 1;

    pass2_naggs = len(x[:n_row][np.where(x >=0 )])+len(x[:n_row][np.where(x < 0 )])


    # Pass 3
    for i in range(n_row):
        xi = x[i]

        if xi != 0: #node i has been aggregated
            if xi > 0:
                x[i] = xi - 1;
            elif xi == -n_row:
                x[i] = -1;
            else:
                x[i] = -xi - 1;
            continue;

        # node i has not been aggregated
        row_start = Ap[i]
        row_end   = Ap[i+1]

        x[i] = next_aggregate
        y[next_aggregate] = i   #y stores a list of the Cpts


        for jj in range(row_start, row_end):
            j = Aj[jj]
            if x[j] == 0: #unmarked neighbors
                x[j] = next_aggregate
        next_aggregate+=1;

    if coord is not None:
        for agg in range(np.max(x)+1):
            agg_mask = np.where(x == agg)
            axs[2].scatter(coord[agg_mask,0], coord[agg_mask,1], s=50, c=colors[(agg+1)%len(colors)]);

    agg_sizes = np.bincount(x[np.where(x >=0 )])
    std = np.std(agg_sizes)
    #print('ave_agg_size %.2f +/- %.2f' % (np.mean(agg_sizes), std))

    return next_aggregate, x, y, pass1_naggs, pass2_naggs;


