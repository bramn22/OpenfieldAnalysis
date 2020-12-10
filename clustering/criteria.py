
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from sklearn.metrics import v_measure_score, adjusted_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def information_analysis(trial_labels, trial_mouselines):
    """ Estimate statistical dependence between response clusters and stimuli"""
    label_counts = np.unique(trial_labels, return_counts=True)
    label_counts = dict(zip(label_counts[0], label_counts[1]))
    P_g = {cluster: count /len(trial_labels) for cluster, count in label_counts.items()}

    line_counts = np.unique(trial_mouselines, return_counts=True)
    line_counts = dict(zip(line_counts[0], line_counts[1]))
    P_s = {line: count /len(trial_mouselines) for line, count in line_counts.items()}

    label_line_counts = np.unique(list(zip(trial_labels, trial_mouselines)), return_counts=True, axis=0)
    #     print(label_line_counts)
    label_line_counts = {(label_line_counts[0][i ,0], label_line_counts[0][i ,1]): label_line_counts[1][i] for i in range(len(label_line_counts[1]))}
    #     label_line_counts = dict(zip(label_line_counts[0], label_line_counts[1]))
    P_gs = {(cluster, line): count /len(trial_labels) for (cluster, line), count in label_line_counts.items()}

    P_g_given_s = {(cluster, line): p/ P_s[line] for (cluster, line), p in P_gs.items()}

    H_g = -sum([p * np.log2(p) for p in P_g.values()])
    H_g_given_s = -sum([P_gs[(cluster, line)] * np.log2(P_g_given_s[(cluster, line)]) for cluster, line in P_gs.keys()])

    return H_g - H_g_given_s
    # Correct estimates of H_g and H_g_given_s
    # Calculate MI as difference of these corrected estimates

def calc_criteria_ground_truth(labels, trial_mouselines):
    results = {}
    results['mutual_info'] = information_analysis(labels, trial_mouselines)
    results['v_measure'] = v_measure_score(trial_mouselines, labels)
    results['adj_mutual_info'] = adjusted_mutual_info_score(trial_mouselines, labels)
    results['adj_rand'] = adjusted_rand_score(trial_mouselines, labels)
    return results

# def calc_criteria_no_ground_truth(data, labels):
#     # The silhouette_score gives the average value for all the samples.
#     # This gives a perspective into the density and separation of the formed
#     # clusters
#     results = {}
#     results['silhouette'] = silhouette_score(data, labels)
#     results['silhouette_samples']
#     return results


def silhouette_analysis(data, labels, detailed=False):
    # Check: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(data, labels)
    if detailed:
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(data, labels)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax.set_xlim([-0.1, 1])
        n_clusters = len(np.unique(labels))
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax.set_ylim([0, len(data) + (n_clusters + 1) * 10])
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title("The silhouette plot for the various clusters.")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.show()
    return silhouette_avg

def davies_bouldin(data, labels):
    return davies_bouldin_score(data, labels)