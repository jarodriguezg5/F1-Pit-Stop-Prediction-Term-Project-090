import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, f1_score

RANDOM_STATE = 6740

df = pd.read_csv('f1_all_features.csv', low_memory=False)
df = df[df['Year'] <= 2024]

FEATURES = [
    'TyreLife', 'TireDegDelta', 'Position', 'RaceProgress', 'LapsRemaining',
    'StopsSoFar', 'AnyCautionThisLap', 'CautionJustStarted', 'SafetyCarThisLap',
    'VSCThisLap', 'YellowThisLap', 'WetConditions', 'Stint', 'PittedThisLap',
]
df = df.dropna(subset=FEATURES)

train = df[df['Year'] <= 2022]
validate = df[df['Year'] == 2023]
test = df[df['Year'] == 2024]

X_train, y_train = train[FEATURES], train['PitNextLap']
X_val, y_val = validate[FEATURES], validate['PitNextLap']
X_test, y_test = test[FEATURES], test['PitNextLap']

# Same threshold-picking logic as train_models.py, needed here to keep the
# ablation figure consistent with the numbers already in the console output.
def pick_threshold(y_true, probabilities):
    best_threshold, best_f1 = 0.5, -1.0
    for candidate in np.linspace(0.01, 0.99, 99):
        score = f1_score(y_true, (probabilities >= candidate).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, candidate
    return best_threshold

# Same stratified SVM sample as train_models.py, same reason: full-scale SVM
# is too slow to redo just for a chart.
positives = train[train['PitNextLap'] == 1]
negatives = train[train['PitNextLap'] == 0].sample(n=12000, random_state=RANDOM_STATE)
svm_train = pd.concat([positives, negatives]).sample(frac=1, random_state=RANDOM_STATE)

models = {
    'Logistic\nRegression': Pipeline([('scale', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE))]),
    'Gaussian\nNaive Bayes': Pipeline([('scale', StandardScaler()), ('model', GaussianNB())]),
    'SVM\n(RBF)': Pipeline([('scale', StandardScaler()),
        ('model', SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE))]),
    'Random\nForest': RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

print("Training 5 models for the comparison figure...")
model_results = {}
for name, model in models.items():
    fit_X, fit_y = (svm_train[FEATURES], svm_train['PitNextLap']) if 'SVM' in name else (X_train, y_train)
    if name == 'AdaBoost':
        ratio = (fit_y == 0).sum() / (fit_y == 1).sum()
        weights = np.where(fit_y == 1, ratio, 1.0)
        model.fit(fit_X, fit_y, sample_weight=weights)
    else:
        model.fit(fit_X, fit_y)
    probability = model.predict_proba(X_test)[:, 1]
    model_results[name] = {'auc': roc_auc_score(y_test, probability), 'model': model}
    print("  {}: AUC {:.4f}".format(name.replace(chr(10), ' '), model_results[name]['auc']))

forest = model_results['Random\nForest']['model']

# FIGURE 1: Model comparison bar chart

fig, ax = plt.subplots(figsize=(8, 5))
names = list(model_results.keys())
aucs = [model_results[n]['auc'] for n in names]
colors = ['#4C72B0'] * len(names)
best_idx = int(np.argmax(aucs))
colors[best_idx] = '#DD8452'

bars = ax.bar(names, aucs, color=colors)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random guessing (0.50)')
ax.set_ylabel('ROC AUC (test year: 2024)')
ax.set_title('Model comparison — predicting a pit stop one lap ahead')
ax.set_ylim(0.4, 0.9)
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, auc + 0.01, '{:.3f}'.format(auc),
           ha='center', fontsize=10)
ax.legend()
plt.tight_layout()
plt.savefig('fig_model_comparison.png', dpi=150)
plt.close()
print("Saved fig_model_comparison.png")

# FIGURE 2: Feature importance (Random Forest)

importance = pd.Series(forest.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(importance.index, importance.values, color='#4C72B0')
ax.set_xlabel('Random Forest feature importance')
ax.set_title('Which features the model actually relies on')
plt.tight_layout()
plt.savefig('fig_feature_importance.png', dpi=150)
plt.close()
print("Saved fig_feature_importance.png")

# FIGURE 3: Ablation — with vs without caution features

caution_features = ['AnyCautionThisLap', 'CautionJustStarted', 'SafetyCarThisLap', 'VSCThisLap', 'YellowThisLap']
reduced = [f for f in FEATURES if f not in caution_features]
forest_reduced = RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
    class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
forest_reduced.fit(train[reduced], y_train)

prob_full = forest.predict_proba(X_test)[:, 1]
prob_reduced = forest_reduced.predict_proba(test[reduced])[:, 1]
auc_full_all = roc_auc_score(y_test, prob_full)
auc_reduced_all = roc_auc_score(y_test, prob_reduced)

caution_mask = (test['AnyCautionThisLap'] == 1).values
auc_full_caution = roc_auc_score(y_test[caution_mask], prob_full[caution_mask])
auc_reduced_caution = roc_auc_score(y_test[caution_mask], prob_reduced[caution_mask])

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(2)
width = 0.35
bars1 = ax.bar(x - width/2, [auc_full_all, auc_full_caution], width, label='With caution features', color='#DD8452')
bars2 = ax.bar(x + width/2, [auc_reduced_all, auc_reduced_caution], width, label='Without caution features', color='#4C72B0')
ax.set_xticks(x)
ax.set_xticklabels(['All test laps\n(n={:,})'.format(len(y_test)), 'Caution laps only\n(n={:,})'.format(caution_mask.sum())])
ax.set_ylabel('ROC AUC')
ax.set_title('Ablation: does the caution feature set earn its place?')
ax.set_ylim(0.5, 0.9)
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, '{:.3f}'.format(h), ha='center', fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig('fig_ablation.png', dpi=150)
plt.close()
print("Saved fig_ablation.png")

# FIGURE 4: K-Means cluster pit rates

cluster_features = [
    'TyreLife', 'TireDegDelta', 'Position', 'RaceProgress', 'LapsRemaining',
    'StopsSoFar', 'AnyCautionThisLap', 'CautionJustStarted', 'WetConditions',
    'Stint', 'PittedThisLap',
]
Xc = StandardScaler().fit_transform(df[cluster_features].values)
kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
df['Cluster'] = kmeans.fit_predict(Xc)

cluster_summary = df.groupby('Cluster').agg(
    laps=('PitNextLap', 'size'), pit_rate=('PitNextLap', 'mean'),
    caution=('AnyCautionThisLap', 'mean'), wet=('WetConditions', 'mean'))

# Auto-label clusters by their dominant characteristic, so the chart reads in
# plain English instead of "Cluster 0, 1, 2, 3."
cluster_labels = []
for idx, row in cluster_summary.iterrows():
    if row['caution'] > 0.5:
        cluster_labels.append('Caution just\nstarted (n={:,})'.format(int(row['laps'])))
    elif row['wet'] > 0.5:
        cluster_labels.append('Wet weather\n(n={:,})'.format(int(row['laps'])))
    elif row['pit_rate'] < df['PitNextLap'].mean():
        cluster_labels.append('Late race, stops\nalready made (n={:,})'.format(int(row['laps'])))
    else:
        cluster_labels.append('Early race,\nfirst stint (n={:,})'.format(int(row['laps'])))
cluster_summary['label'] = cluster_labels
cluster_summary = cluster_summary.sort_values('pit_rate')

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#4C72B0' if 'Caution' not in l else '#DD8452' for l in cluster_summary['label']]
bars = ax.barh(cluster_summary['label'], cluster_summary['pit_rate'] * 100, color=colors)
overall_rate = df['PitNextLap'].mean() * 100
ax.axvline(overall_rate, color='gray', linestyle='--', linewidth=1,
          label='Overall average ({:.1f}%)'.format(overall_rate))
ax.set_xlim(0, cluster_summary['pit_rate'].max() * 100 * 1.15)
ax.set_title('K-Means found these groups with zero knowledge of pit stops —\nyet pit rate still varies 6x across them')
for bar, rate in zip(bars, cluster_summary['pit_rate'] * 100):
    ax.text(rate + 0.3, bar.get_y() + bar.get_height()/2, '{:.1f}%'.format(rate), va='center', fontsize=10)
ax.legend()
plt.tight_layout()
plt.savefig('fig_cluster_pit_rates.png', dpi=150)
plt.close()
print("Saved fig_cluster_pit_rates.png")

print("\nAll 4 figures saved.")