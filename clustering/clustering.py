from sklearn.cluster import KMeans, AgglomerativeClustering
from clustering.criteria import calc_criteria_ground_truth, silhouette_analysis, davies_bouldin
from tqdm import tqdm
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import AgglomerativeClustering

def run_analysis_kmeans(matrix, trial_mouselines, silhouette=True):
    results = {}
    for k in tqdm(range(2, 15)):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100)
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

def run_analysis_gaussian_mixture(matrix, trial_mouselines):
    results = {}

    for k in tqdm(range(2, 15)):
        mixt = GaussianMixture(n_components=k, n_init=50)
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

def run_kmeans(matrix, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=10000)
    kmeans.fit(matrix)
    return kmeans

def run_gaussian_mixture(matrix, n_components=5):
    mixt = GaussianMixture(n_components=n_components, n_init=500)
    mixt.fit(matrix)
    return mixt

def run_bayesian_gaussian_mixture(matrix):
    mixt = BayesianGaussianMixture(15) # will end up using 15 or less clusters!
    mixt.fit(matrix)
    return mixt

def run_agglomerative(matrix, n_clusters):
    mixt = AgglomerativeClustering(n_clusters=n_clusters)
    return mixt.fit_predict(matrix)