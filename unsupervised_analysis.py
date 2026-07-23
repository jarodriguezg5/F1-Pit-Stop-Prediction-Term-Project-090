import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import KernelDensity
from sklearn.metrics import silhouette_score

RANDOM_STATE = 6740

df = pd.read_csv('f1_2024_features.csv')

# Same feature set as the supervised models, minus the three individual flag
# columns (SafetyCarThisLap, VSCThisLap, YellowThisLap) that AnyCautionThisLap
# already summarizes. PCA reads correlated duplicate columns as one inflated
# signal, so trimming them keeps the components honest.
FEATURES = [
    'TyreLife', 'TireDegDelta', 'Position', 'RaceProgress', 'LapsRemaining',
    'StopsSoFar', 'AnyCautionThisLap', 'CautionJustStarted', 'WetConditions',
    'Stint', 'PittedThisLap',
]

# The point of everything below: none of it ever looks at PitNextLap while
# fitting. We only bring the label back in at the very end, to check whether
# structure the algorithms found on their own lines up with real pit stops.
X = df[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Unsupervised analysis on {} laps, {} features".format(len(df), len(FEATURES)))
print("(PitNextLap is not used anywhere in this section except the final checks)")

# PCA

# PCA finds new axes that capture the most spread in the data, ranked by how
# much variance each one explains. It's mainly here so we can actually SEE an
# 11-dimensional feature space - impossible to plot directly, easy in 2D.
pca_full = PCA(random_state=RANDOM_STATE).fit(X_scaled)
cumulative = np.cumsum(pca_full.explained_variance_ratio_)

print("\nPCA - cumulative variance explained by component count:")
for i, c in enumerate(cumulative, 1):
    print("  {} components: {:.1%}".format(i, c))

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
print("\nFirst 2 components explain {:.1%} of total variance".format(
    pca.explained_variance_ratio_.sum()))

# Loadings show which original features each component is built from. Reading
# these tells us what PC1 and PC2 actually MEAN in racing terms.
loadings = pd.DataFrame(pca.components_.T, index=FEATURES, columns=['PC1', 'PC2'])
print("\nWhat each component is made of (sorted by strength on PC1):")
print(loadings.reindex(loadings['PC1'].abs().sort_values(ascending=False).index))

# CHOOSING K FOR K-MEANS

# K-Means needs to be told how many clusters to find. Silhouette score checks
# how well-separated the clusters are for a given k (higher is better,
# range -1 to 1). We scan a range of k and let the data suggest a number,
# rather than picking one arbitrarily.
#
# Silhouette itself is evaluated on a random sample - computing it on the full
# 26k rows compares every point to every other point, which is slow. The
# clustering itself still uses all the data; only the score check is sampled.
rng = np.random.RandomState(RANDOM_STATE)
sample_idx = rng.choice(len(X_scaled), min(5000, len(X_scaled)), replace=False)

print("\nK-Means - scanning cluster counts:")
print("{:>3} {:>12} {:>12}".format("k", "silhouette", "inertia"))
for k in range(2, 8):
    km_test = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels_test = km_test.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled[sample_idx], labels_test[sample_idx])
    print("{:>3} {:>12.4f} {:>12.0f}".format(k, sil, km_test.inertia_))

# K-MEANS

# k=4 gives a clear silhouette peak relative to its neighbors and produces
# clusters simple enough to describe in plain language - both matter for a
# report, not just the number on its own.
K = 4
kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
df['KMeansCluster'] = kmeans.fit_predict(X_scaled)

print("\nK-Means (k={}) cluster sizes:".format(K))
print(df['KMeansCluster'].value_counts().sort_index().to_string())

# Cluster centers, translated back into real units (laps, positions, etc.)
# rather than the scaled numbers K-Means actually works with. This is what
# lets us name each cluster something meaningful instead of "Cluster 2."
print("\nWhat each cluster looks like (average feature values):")
print(df.groupby('KMeansCluster')[FEATURES].mean().T.round(3))

# THE KEY CHECK: does pit rate differ across clusters we built blind?

# This is the payoff. The clusters were built with zero knowledge of
# PitNextLap. If pit rate still comes out very different across them, that
# means the SAME features driving the supervised models also describe
# real structure in the data on their own - independent evidence the feature
# set captures something genuine about race strategy, not just noise the
# supervised models happened to fit.
print("\nPit rate on the NEXT lap, by K-Means cluster (label never used to build these):")
print(df.groupby('KMeansCluster')['PitNextLap'].agg(laps='size', pit_rate='mean').to_string())

# SPECTRAL CLUSTERING

# K-Means assumes clusters are round blobs of similar size. Spectral
# clustering doesn't - it builds a graph connecting nearby points and cuts it
# into groups, so it can find clusters with irregular or elongated shapes that
# K-Means would slice incorrectly.
#
# The trade-off is cost: it needs a similarity comparison between every pair
# of points, which explodes on 26,000 rows. We subsample to keep it running in
# seconds rather than hours - this is a standard, expected trade-off for
# spectral methods on large datasets, worth stating plainly in the report.
spectral_sample_size = 3000
spectral_idx = rng.choice(len(X_scaled), spectral_sample_size, replace=False)
X_spectral = X_scaled[spectral_idx]
df_spectral = df.iloc[spectral_idx].copy()

spectral = SpectralClustering(
    n_clusters=K, random_state=RANDOM_STATE,
    affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans')
df_spectral['SpectralCluster'] = spectral.fit_predict(X_spectral)

print("\nSpectral Clustering (n={} sample, k={}) cluster sizes:".format(
    spectral_sample_size, K))
print(df_spectral['SpectralCluster'].value_counts().sort_index().to_string())

print("\nPit rate on the NEXT lap, by Spectral cluster:")
print(df_spectral.groupby('SpectralCluster')['PitNextLap'].agg(
    laps='size', pit_rate='mean').to_string())

# DENSITY ESTIMATION

# Kernel Density Estimation maps out where race-states are common versus rare
# in the feature space, without assuming any fixed number of groups the way
# clustering does. It answers a different question than clustering: not "which
# group does this lap belong to" but "how typical is this lap's situation?"
#
# Fit in the same 2D PCA space we already built, so the density map lines up
# with the same plot as the clusters.
kde = KernelDensity(bandwidth=0.3, kernel='gaussian')
kde.fit(X_pca)
df['LogDensity'] = kde.score_samples(X_pca)

# Split into quartiles: Q1 = rarest, most unusual race states. Q4 = most
# common, everyday race states.
df['DensityQuartile'] = pd.qcut(
    df['LogDensity'], 4, labels=['Q1 rarest', 'Q2', 'Q3', 'Q4 most common'])

print("\nPit rate on the NEXT lap, by density quartile:")
print(df.groupby('DensityQuartile', observed=True)['PitNextLap'].agg(
    laps='size', pit_rate='mean').to_string())

# FIGURE: PCA scatter colored by K-Means cluster

# Saved as a PNG for the report - drop it straight into the Overleaf root and
# reference it with \includegraphics, same as the other figures.
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['KMeansCluster'],
                     cmap='tab10', s=3, alpha=0.4)
ax.set_xlabel('PC1 ({:.1%} variance)'.format(pca.explained_variance_ratio_[0]))
ax.set_ylabel('PC2 ({:.1%} variance)'.format(pca.explained_variance_ratio_[1]))
ax.set_title('K-Means clusters (k={}) in PCA-reduced feature space'.format(K))
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('pca_clusters.png', dpi=150)
print("\nSaved figure to pca_clusters.png")