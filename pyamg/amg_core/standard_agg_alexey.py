import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def standard_aggregation_py(n_row, Ap, Aj, x, y, coord=None, modify=False):
    #print('standard_aggregation_py')

    if coord is not None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        fig, axs   = plt.subplots(ncols=(3), figsize=(6*(3), 6))
        for i, ax in enumerate(axs):
            triplot(coord[1], axes=ax, interior_kw=dict(alpha=0.2, linewidth=2/(3)),
                    boundary_kw=dict(colors=['k']*4, linewidths=[1./(3)]*4))
            ax.scatter(coord[0][:,0], coord[0][:,1], color='k', s=3);
            ax.set_title('Phase %d' % (i+1))
        coord = coord[0]

    x[:n_row] = 0
    next_aggregate = 1; # number of aggregates + 1

    aggs_dict = {}
    # Pass 1
    for i in range(n_row):
        if x[i]:
            continue # alreay marked
        else:
            aggs_dict[next_aggregate] = []

        row_start = Ap[i]
        row_end   = Ap[i+1]
        #Determine whether all neighbors of this node are free (not already aggregates)
        has_aggregated_neighbors = False
        has_neighbors = False

        for jj in range(row_start, row_end):
            j = Aj[jj]
            if i != j:
                has_neighbors = True
                if x[j]:
                    has_aggregated_neighbors = True
                    break

        if not has_neighbors:  #isolated node, do not aggregate
            x[i] = -n_row;
        elif not has_aggregated_neighbors: #Make an aggregate out of this node and its neighbors
            x[i] = next_aggregate
            aggs_dict[next_aggregate].append(i)
            y[next_aggregate-1] = i # y stores a list of the Cpts

            for jj in range(row_start, row_end):
                x[Aj[jj]] = next_aggregate;
                aggs_dict[next_aggregate].append(Aj[jj])
            next_aggregate+=1;

    #print('negative values - isolated nodes (not agg-ed); \n postive values - not agg-ed and neighbors are not either; \n zeros - non-agg-ed nodes')
    #print('Phase 1:\n', x)
    if coord is not None:
        for agg in range(1,np.max(x)+1):
            agg_mask = np.where(x == agg)
            axs[0].scatter(coord[agg_mask,0], coord[agg_mask,1], s=50, c=colors[agg%len(colors)]);



    # Pass 2: Add unaggregated nodes to any neighboring aggregate
    if not modify:
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

        mean = 0.0
        for key in aggs_dict.keys():
            aggs_dict[key] = len(aggs_dict[key])
            mean += aggs_dict[key]
        mean /= len(aggs_dict.keys())
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

            countss = np.bincount(neighbors)

            while np.sum(countss):
                agg_ids.append(np.argmax(countss))
                conn_to_agg.append(countss[agg_ids[-1]])
                countss[agg_ids[-1]] = 0
                agg_size.append(aggs_dict[agg_ids[-1]])


            m = (np.array(agg_size) - mean)
            m[np.where(m > 0)] = 1
            m = np.abs(m)
            score = np.array(agg_size)*np.array(conn_to_agg)*m
            pick = agg_ids[np.argmax(score)]

            x[i] = -pick
            #x[i] = -agg_ids[0];


    next_aggregate-= 1;

    if coord is not None:
        #print('Phase 2:\n', x)
        for agg in range(np.max(x)+1):
            agg_mask = np.where(x == agg)
            axs[1].scatter(coord[agg_mask,0], coord[agg_mask,1], s=50, c=colors[agg%len(colors)]);

    #    agg_mask = np.where(x == -5)
    #    axs[1].scatter(coord[agg_mask,0], coord[agg_mask,1], s=70, c=colors[i%len(colors)]);


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
        #print('Phase 3:\n', x)
        for agg in range(np.max(x)+1):
            agg_mask = np.where(x == agg)
            axs[2].scatter(coord[agg_mask,0], coord[agg_mask,1], s=50, c=colors[(agg+1)%len(colors)]);

    agg_sizes = np.bincount(x[np.where(x >=0 )])
    std = np.std(agg_sizes)
    print('ave_agg_size %.2f +/- %.2f' % (np.mean(agg_sizes), std))

    return next_aggregate, x, y;


