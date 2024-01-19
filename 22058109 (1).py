import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import cluster_tools as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import errors as err

def read_data(filename):
    """
    Read and manipulate data from the provided file.

    Parameters:
    - filename (str): The path to the data file.

    Returns:
    - df (pd.DataFrame): Original DataFrame.
    """
    df = pd.read_excel(filename, skiprows=[0, 1, 2])
    return df

def clean_data(df, indicator1, indicator2, year):
    """
    Clean and preprocess data for specified indicators and year.

    Parameters:
    - df (pd.DataFrame): Original DataFrame.
    - indicator1 (str): Name of the first indicator.
    - indicator2 (str): Name of the second indicator.
    - year (str): Year of interest.

    Returns:
    - df_clean (pd.DataFrame): Cleaned DataFrame.
    """
    df1 = df[df['Indicator Name'] == indicator1][[year]].reset_index(drop=True)
    df2 = df[df['Indicator Name'] == indicator2][[year]].reset_index(drop=True)
    df_result = pd.concat([df1, df2], axis=1)
    df_result.columns = [indicator1, indicator2]
    df_result = df_result.dropna()

    return df_result
def kmeans_cluster(df_cluster, num_clusters):
    """
    Perform k-means clustering on the given DataFrame.

    Parameters:
    - df_cluster (pd.DataFrame): DataFrame for clustering.
    - num_clusters (int): The number of clusters.

    Returns:
    - cluster_labels (array): Labels of the clusters.
    - cluster_centers (list of lists): Coordinates of the cluster centers.
    """
    kmeans_model = KMeans(n_clusters=num_clusters)
    kmeans_model.fit(df_cluster)
    cluster_labels = kmeans_model.labels_
    cluster_centers = kmeans_model.cluster_centers_

    return cluster_labels, cluster_centers

def print_silhouette_scores(df_cluster, max_clusters):
    """
    Print silhouette scores for different cluster numbers.

    Parameters:
    - df_cluster (pd.DataFrame): DataFrame for clustering.
    - max_clusters (int): The maximum number of clusters to consider.
    """
    print("n   score")
    for ncluster in range(2, max_clusters + 1):
        labels, _ = kmeans_cluster(df_cluster, ncluster)
        score = metrics.silhouette_score(df_cluster, labels)
        print(ncluster, score)

def plot_clustering(df_clean, indicators, labels):
    """
    Plot the clustering based on two indicators.

    Parameters:
    - df_clean (pd.DataFrame): Cleaned DataFrame.
    - indicators (list): List of indicator names.
    - labels (array): Labels of the clusters.
    """
    plt.figure()
    cm = plt.cm.get_cmap('Set1')
    plt.scatter(df_clean[indicators[0]], df_clean[indicators[1]], 10,
                labels, marker='o', cmap=cm)
    plt.xlabel(indicators[0])
    plt.ylabel(indicators[1])
    plt.title(f"{indicators[0]} vs. {indicators[1]}")
    plt.show()

def fit_and_plot(df, year):
    """
    Fit a polynomial curve to the data and plot it with confidence range.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - year (int): Year for which to forecast.

    Returns:
    - None
    """
    def poly(x, *params):
        return np.polyval(params, x)

    def err_ranges(x, func, params, sigma):
        y_vals = func(x, *params)
        delta = func(x, *params + sigma) - y_vals
        return y_vals - delta, y_vals + delta

    df = df.reset_index()
    df["Cereal yield (kg per hectare)"] = pd.to_numeric(df["Cereal yield (kg per hectare)"])
    df["Country Name"] = pd.to_numeric(df["Country Name"])

    param, covar = opt.curve_fit(poly, df["Country Name"], df["Cereal yield (kg per hectare)"])
    sigma = np.sqrt(np.diag(covar))
    forecast = poly(year, *param)
    low, up = err_ranges(df["Country Name"], poly, param, sigma)
    df["fit1"] = poly(df["Country Name"], *param)

    plt.figure()
    plt.plot(df["Country Name"], df["Cereal yield (kg per hectare)"], label="yield", c='blue')
    plt.plot(year, forecast, label="forecast", c='red')
    plt.fill_between(year, low, up, color="yellow", alpha=0.8)
    plt.xlabel("Country", fontsize=16)
    plt.ylabel("Yield", fontsize=14)
    plt.title("Country as per yield", fontsize=18)
    plt.legend()
    plt.show()

def main():
    filename = "API_19_DS2_en_excel_v2_6002116.xls"
    indicators = ['Cereal yield (kg per hectare)', 'Access to electricity (% of population)']
    year = '2012'

    # Read data
    df = read_data(filename)

    # Clean data
    df_clean = clean_data(df, indicators[0], indicators[1], year)

    pd.plotting.scatter_matrix(df_clean)

    # Perform k-means clustering
    num_clusters = 3
    labels, centers = kmeans_cluster(df_clean, num_clusters)

    # Print silhouette scores for different cluster numbers
    print_silhouette_scores(df_clean, max_clusters=9)

    # Plot the clustering
    plot_clustering(df_clean, indicators, labels)
     # Fit and plot
    fit_and_plot(df, year)

if __name__ == "__main__":
    main()