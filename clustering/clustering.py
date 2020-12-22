from sklearn.cluster import KMeans, AgglomerativeClustering
from clustering.criteria import calc_criteria_ground_truth, silhouette_analysis, davies_bouldin
from tqdm import tqdm
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids

def run_analysis_kmeans(matrix, trial_mouselines, k_range=None, n_init=200, max_iter=500, init='random'):
    results = {}
    if None:
        k_range = range(2, 15)
    for k in tqdm(k_range):
        kmeans = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter)
        kmeans.fit(matrix)
        labels = kmeans.predict(matrix)
        k_results = calc_criteria_ground_truth(labels, trial_mouselines)
        k_results['ssq'] = kmeans.inertia_
        k_results['silhouette'] = silhouette_analysis(matrix, labels)
        k_results['davies_bouldin'] = davies_bouldin(matrix, labels)
        if results == {}:
            for crit, res in k_results.items():
                results[crit] = [res]
        else:
            for crit, res in k_results.items():
                results[crit].append(res)

    return results

def run_analysis_gaussian_mixture(matrix, trial_mouselines, n_init=200, k_range=None, max_iter=500, init='random'):
    results = {}
    if None:
        k_range = range(2, 15)
    for k in tqdm(k_range):
        mixt = GaussianMixture(n_components=k, n_init=n_init, max_iter=max_iter, init_params=init)
        mixt.fit(matrix)

        labels = mixt.predict(matrix)
        k_results = calc_criteria_ground_truth(labels, trial_mouselines)
        k_results['aic'] = mixt.aic(matrix)
        k_results['bic'] = mixt.bic(matrix)
        k_results['silhouette'] = silhouette_analysis(matrix, labels)
        k_results['davies_bouldin'] = davies_bouldin(matrix, labels)

        if results == {}:
            for crit, res in k_results.items():
                results[crit] = [res]
        else:
            for crit, res in k_results.items():
                results[crit].append(res)

    return results

def run_analysis_agglomerative(matrix, trial_mouselines):
    results = {}

    for k in tqdm(range(2, 20)):
        mixt = AgglomerativeClustering(n_clusters=k)

        labels = mixt.fit_predict(matrix)
        k_results = calc_criteria_ground_truth(labels, trial_mouselines)
        k_results['silhouette'] = silhouette_analysis(matrix, labels)
        k_results['davies_bouldin'] = davies_bouldin(matrix, labels)

        if results == {}:
            for crit, res in k_results.items():
                results[crit] = [res]
        else:
            for crit, res in k_results.items():
                results[crit].append(res)

    return results

def run_analysis_kmedoids(matrix, trial_mouselines, k_range=None, init='random', n_init=100, metric='euclidean', max_iter=300):
    results = {}
    if None:
        k_range = range(2, 15)
    for k in tqdm(k_range):
        k_results = None
        min_inertia = float('inf')
        for i in range(n_init):
            kmedoids = KMedoids(n_clusters=k, init=init, metric=metric, max_iter=max_iter)
            kmedoids.fit(matrix)
            if kmedoids.inertia_ < min_inertia:
                min_inertia = kmedoids.inertia_
                labels = kmedoids.predict(matrix)
                k_results = calc_criteria_ground_truth(labels, trial_mouselines)
                k_results['ssq'] = kmedoids.inertia_
                k_results['silhouette'] = silhouette_analysis(matrix, labels)
                k_results['davies_bouldin'] = davies_bouldin(matrix, labels)
        if results == {}:
            for crit, res in k_results.items():
                results[crit] = [res]
        else:
            for crit, res in k_results.items():
                results[crit].append(res)

    return results

def run_kmeans(matrix, n_clusters=5, n_init=1000, max_iter=1000, init='random'):
    kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=n_init)
    kmeans.fit(matrix)
    return kmeans

def run_kmedoids(matrix, n_clusters=5, n_init=1000, metric='euclidean', max_iter=1000, init='random'):
    best_kmedoids = None
    min_inertia = float('inf')
    for i in range(n_init):
        kmedoids = KMedoids(n_clusters=n_clusters, init=init, max_iter=max_iter, metric=metric)
        kmedoids.fit(matrix)
        if kmedoids.inertia_ < min_inertia:
            min_inertia = kmedoids.inertia_
            best_kmedoids = kmedoids
    return best_kmedoids

def run_gaussian_mixture(matrix, n_components=5, n_init=1000, max_iter=100, init='random'):
    mixt = GaussianMixture(n_components=n_components, n_init=n_init, max_iter=max_iter, init_params=init)
    mixt.fit(matrix)
    return mixt

def run_bayesian_gaussian_mixture(matrix):
    mixt = BayesianGaussianMixture(15) # will end up using 15 or less clusters!
    mixt.fit(matrix)
    return mixt

def run_agglomerative(matrix, n_clusters):
    mixt = AgglomerativeClustering(n_clusters=n_clusters)
    return mixt.fit_predict(matrix)