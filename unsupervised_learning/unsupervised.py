import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import KernelDensity
from sklearn.metrics import silhouette_score

RANDOM_STATE = 6740

# Same root-anchoring trick as the other scripts.
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Data" / "final_data" / "f1_all_features.csv"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'TyreLife', 'TireDegDelta', 'Position', 'RaceProgress', 'LapsRemaining',
    'StopsSoFar', 'AnyCautionThisLap', 'CautionJustStarted', 'WetConditions',
    'Stint', 'PittedThisLap',
]


def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df['Year'] <= 2024]

    # Same 904-row gap as supervised.py - a handful of early races missing
    # TyreLife or Stint. PCA and K-Means can't run with blanks in the input.
    before = len(df)
    df = df.dropna(subset=FEATURES)
    print("Dropped {} rows with missing features".format(before - len(df)))

    return df


def run_pca(X_scaled):
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

    return pca, X_pca


def scan_k(X_scaled, rng):
    sample_idx = rng.choice(len(X_scaled), 5000, replace=False)

    print("\nK-Means - scanning cluster counts:")
    print("{:>3} {:>12} {:>12}".format("k", "silhouette", "inertia"))
    for k in range(2, 8):
        km_test = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels_test = km_test.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled[sample_idx], labels_test[sample_idx])
        print("{:>3} {:>12.4f} {:>12.0f}".format(k, sil, km_test.inertia_))


def run_kmeans(df, X_scaled, k=4):
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    df['KMeansCluster'] = kmeans.fit_predict(X_scaled)

    print("\nK-Means (k={}) cluster sizes:".format(k))
    print(df['KMeansCluster'].value_counts().sort_index().to_string())

    print("\nWhat each cluster looks like (average feature values):")
    print(df.groupby('KMeansCluster')[FEATURES].mean().T.round(3))

    print("\nPit rate on the NEXT lap, by K-Means cluster (label never used to build these):")
    print(df.groupby('KMeansCluster')['PitNextLap'].agg(laps='size', pit_rate='mean').to_string())

    return df, kmeans


def run_spectral(df, X_scaled, rng, k=4, sample_size=3000):
    # Same subsampling logic as before, same reason: spectral clustering
    # compares every point to every other point, which is unworkable at full scale.
    spectral_idx = rng.choice(len(X_scaled), sample_size, replace=False)
    X_spectral = X_scaled[spectral_idx]
    df_spectral = df.iloc[spectral_idx].copy()

    spectral = SpectralClustering(
        n_clusters=k, random_state=RANDOM_STATE,
        affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans')
    df_spectral['SpectralCluster'] = spectral.fit_predict(X_spectral)

    print("\nSpectral Clustering (n={} sample, k={}) cluster sizes:".format(sample_size, k))
    print(df_spectral['SpectralCluster'].value_counts().sort_index().to_string())

    print("\nPit rate on the NEXT lap, by Spectral cluster:")
    print(df_spectral.groupby('SpectralCluster')['PitNextLap'].agg(
        laps='size', pit_rate='mean').to_string())


def run_kde(df, X_pca, rng, fit_size=4000):
    # KDE's cost scales with the number of points it's FIT on, since scoring a
    # new point means comparing it against every point in the fitted model.
    # Fitting on a 4,000-row sample and scoring every lap against that smaller
    # reference set keeps this fast with no real change to which laps come
    # out "rare" vs "common."
    kde_fit_idx = rng.choice(len(X_pca), fit_size, replace=False)
    kde = KernelDensity(bandwidth=0.3, kernel='gaussian')
    kde.fit(X_pca[kde_fit_idx])
    df['LogDensity'] = kde.score_samples(X_pca)

    df['DensityQuartile'] = pd.qcut(
        df['LogDensity'], 4, labels=['Q1 rarest', 'Q2', 'Q3', 'Q4 most common'])

    print("\nPit rate on the NEXT lap, by density quartile:")
    print(df.groupby('DensityQuartile', observed=True)['PitNextLap'].agg(
        laps='size', pit_rate='mean').to_string())

    return df


# FIGURE 1: PCA scatter colored by K-Means cluster

def make_pca_scatter_figure(df, X_pca, pca, rng, k, sample_size=15000):
    # Plotting every lap would make an unreadably dense, slow-to-render blob.
    # A random sample shows the same shape and cluster separation while
    # staying legible and quick to save.
    plot_idx = rng.choice(len(X_pca), sample_size, replace=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[plot_idx, 0], X_pca[plot_idx, 1],
                         c=df['KMeansCluster'].values[plot_idx],
                         cmap='tab10', s=3, alpha=0.4)
    ax.set_xlabel('PC1 ({:.1%} variance)'.format(pca.explained_variance_ratio_[0]))
    ax.set_ylabel('PC2 ({:.1%} variance)'.format(pca.explained_variance_ratio_[1]))
    ax.set_title('K-Means clusters (k={}) in PCA-reduced feature space\n'
                 '2018-2024, {} laps shown ({:,}-point sample)'.format(
                     k, len(plot_idx), sample_size))
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_clusters.png", dpi=150)
    plt.close()
    print("\nSaved {}".format(OUTPUT_DIR / "pca_clusters.png"))


# FIGURE 2: K-Means cluster pit rates, auto-labeled in plain English

def make_cluster_pit_rate_figure(df):
    cluster_summary = df.groupby('KMeansCluster').agg(
        laps=('PitNextLap', 'size'), pit_rate=('PitNextLap', 'mean'),
        caution=('AnyCautionThisLap', 'mean'), wet=('WetConditions', 'mean'))

    overall_rate = df['PitNextLap'].mean()

    # Auto-label clusters by their dominant characteristic, so the chart reads
    # in plain English instead of "Cluster 0, 1, 2, 3."
    cluster_labels = []
    for idx, row in cluster_summary.iterrows():
        if row['caution'] > 0.5:
            cluster_labels.append('Caution just\nstarted (n={:,})'.format(int(row['laps'])))
        elif row['wet'] > 0.5:
            cluster_labels.append('Wet weather\n(n={:,})'.format(int(row['laps'])))
        elif row['pit_rate'] < overall_rate:
            cluster_labels.append('Late race, stops\nalready made (n={:,})'.format(int(row['laps'])))
        else:
            cluster_labels.append('Early race,\nfirst stint (n={:,})'.format(int(row['laps'])))
    cluster_summary['label'] = cluster_labels
    cluster_summary = cluster_summary.sort_values('pit_rate')

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#4C72B0' if 'Caution' not in l else '#DD8452' for l in cluster_summary['label']]
    bars = ax.barh(cluster_summary['label'], cluster_summary['pit_rate'] * 100, color=colors)
    ax.axvline(overall_rate * 100, color='gray', linestyle='--', linewidth=1,
              label='Overall average ({:.1f}%)'.format(overall_rate * 100))
    ax.set_xlabel('Pit rate on the NEXT lap (%)')
    ax.set_title('K-Means found these groups with zero knowledge of pit stops —\n'
                'yet pit rate still varies across them')
    for bar, rate in zip(bars, cluster_summary['pit_rate'] * 100):
        ax.text(rate + 0.3, bar.get_y() + bar.get_height()/2, '{:.1f}%'.format(rate),
               va='center', fontsize=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_cluster_pit_rates.png", dpi=150)
    plt.close()
    print("Saved {}".format(OUTPUT_DIR / "fig_cluster_pit_rates.png"))


def main():
    df = load_data()

    X = df[FEATURES].values
    X_scaled = StandardScaler().fit_transform(X)

    print("Unsupervised analysis on {} laps, {} features".format(len(df), len(FEATURES)))
    print("(PitNextLap is not used anywhere in this section except the final checks)")

    pca, X_pca = run_pca(X_scaled)

    rng = np.random.RandomState(RANDOM_STATE)
    scan_k(X_scaled, rng)

    K = 4
    df, kmeans = run_kmeans(df, X_scaled, k=K)
    run_spectral(df, X_scaled, rng, k=K)
    df = run_kde(df, X_pca, rng)

    make_pca_scatter_figure(df, X_pca, pca, rng, k=K)
    make_cluster_pit_rate_figure(df)

    print("\nAll unsupervised figures saved to {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()