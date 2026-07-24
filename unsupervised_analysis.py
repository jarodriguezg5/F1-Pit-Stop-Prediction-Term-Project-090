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

df = pd.read_csv('f1_all_features.csv', low_memory=False)

# Same as train_models.py - 2025 is one incomplete race, drop it.
df = df[df['Year'] <= 2024]

FEATURES = [
    'TyreLife', 'TireDegDelta', 'Position', 'RaceProgress', 'LapsRemaining',
    'StopsSoFar', 'AnyCautionThisLap', 'CautionJustStarted', 'WetConditions',
    'Stint', 'PittedThisLap',
]

# Same 904-row gap as the supervised script - a handful of early races missing
# TyreLife or Stint. PCA and K-Means can't run with blanks in the input, so
# these rows get dropped here too.
before = len(df)
df = df.dropna(subset=FEATURES)
print("Dropped {} rows with missing features".format(before - len(df)))

X = df[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Unsupervised analysis on {} laps, {} features".format(len(df), len(FEATURES)))
print("(PitNextLap is not used anywhere in this section except the final checks)")

# PCA

pca_full = PCA(random_state=RANDOM_STATE).fit(X_scaled)
cumulative = np.cumsum(pca_full.explained_variance_ratio_)

print("\nPCA - cumulative variance explained by component count:")
for i, c in enumerate(cumulative, 1):
    print("  {} components: {:.1%}".format(i, c))

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
print("\nFirst 2 components explain {:.1%} of total variance".format(
    pca.explained_variance_ratio_.sum()))

loadings = pd.DataFrame(pca.components_.T, index=FEATURES, columns=['PC1', 'PC2'])
print("\nWhat each component is made of (sorted by strength on PC1):")
print(loadings.reindex(loadings['PC1'].abs().sort_values(ascending=False).index))

# CHOOSING K FOR K-MEANS

rng = np.random.RandomState(RANDOM_STATE)
sample_idx = rng.choice(len(X_scaled), 5000, replace=False)

print("\nK-Means - scanning cluster counts:")
print("{:>3} {:>12} {:>12}".format("k", "silhouette", "inertia"))
for k in range(2, 8):
    km_test = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels_test = km_test.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled[sample_idx], labels_test[sample_idx])
    print("{:>3} {:>12.4f} {:>12.0f}".format(k, sil, km_test.inertia_))

# K-MEANS

K = 4
kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
df['KMeansCluster'] = kmeans.fit_predict(X_scaled)

print("\nK-Means (k={}) cluster sizes:".format(K))
print(df['KMeansCluster'].value_counts().sort_index().to_string())

print("\nWhat each cluster looks like (average feature values):")
print(df.groupby('KMeansCluster')[FEATURES].mean().T.round(3))

print("\nPit rate on the NEXT lap, by K-Means cluster (label never used to build these):")
print(df.groupby('KMeansCluster')['PitNextLap'].agg(laps='size', pit_rate='mean').to_string())

# SPECTRAL CLUSTERING

# Same subsampling logic as before, same reason: spectral clustering compares
# every point to every other point, which is unworkable at 158,000 rows.
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

# KDE's cost scales with the number of points it's FIT on, since scoring a new
# point means comparing it against every point in the fitted model. Fitting on
# all 158,000 rows made this take upwards of 10 minutes; fitting on a 4,000-row
# sample and then scoring every lap against that smaller reference set brings
# it under 30 seconds, with no real change to which laps come out "rare" vs
# "common" - the shape of the density landscape is what matters, and 4,000
# points is plenty to trace it out in 2D.
kde_fit_idx = rng.choice(len(X_pca), 4000, replace=False)
kde = KernelDensity(bandwidth=0.3, kernel='gaussian')
kde.fit(X_pca[kde_fit_idx])
df['LogDensity'] = kde.score_samples(X_pca)

df['DensityQuartile'] = pd.qcut(
    df['LogDensity'], 4, labels=['Q1 rarest', 'Q2', 'Q3', 'Q4 most common'])

print("\nPit rate on the NEXT lap, by density quartile:")
print(df.groupby('DensityQuartile', observed=True)['PitNextLap'].agg(
    laps='size', pit_rate='mean').to_string())

# FIGURE: PCA scatter colored by K-Means cluster

# Plotting all 158,000 points would make an unreadably dense, slow-to-render
# blob. A 15,000-point random sample shows the same shape and cluster
# separation while staying legible and quick to save.
plot_idx = rng.choice(len(X_pca), 15000, replace=False)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[plot_idx, 0], X_pca[plot_idx, 1],
                     c=df['KMeansCluster'].values[plot_idx],
                     cmap='tab10', s=3, alpha=0.4)
ax.set_xlabel('PC1 ({:.1%} variance)'.format(pca.explained_variance_ratio_[0]))
ax.set_ylabel('PC2 ({:.1%} variance)'.format(pca.explained_variance_ratio_[1]))
ax.set_title('K-Means clusters (k={}) in PCA-reduced feature space\n'
             '2018-2024, {} laps shown (15,000-point sample)'.format(K, len(plot_idx)))
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('pca_clusters.png', dpi=150)
print("\nSaved figure to pca_clusters.png")