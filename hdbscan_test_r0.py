import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
#matplotlib inline

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}
plt.rcParams["figure.figsize"] = [9,7]
"""!pip install hdbscan"""
import hdbscan

num=100
moons, _ = data.make_moons(n_samples=num, noise=0.01)
blobs, _ = data.make_blobs(n_samples=num, centers=[(-0.75,2.25), (1.0, -2.0)], cluster_std=0.25)
blobs2, _ = data.make_blobs(n_samples=num, centers=[(2,2.25), (-1, -2.0)], cluster_std=0.4)
test_data = np.vstack([moons, blobs,blobs2])
plt.figure(num=1)
plt.scatter(test_data.T[0], test_data.T[1], color='b', **plot_kwds)
#plt.show()

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(test_data)

plt.figure(num=2)
clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=10, edge_linewidth=2)
#plt.show()


plt.figure(num=3)
clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)

plt.figure(num=4)
clusterer.condensed_tree_.plot()

plt.figure(num=5)
clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())

plt.figure(num=6)
palette = sns.color_palette()
cluster_colors = [sns.desaturate(palette[col], sat)
                  if col >= 0 else (0.3,0.3,0.3) for col, sat in
                  zip(clusterer.labels_, clusterer.probabilities_)]
plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
plt.show()
