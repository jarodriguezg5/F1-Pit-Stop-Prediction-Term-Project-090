import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score)

from evaluation import RANDOM_STATE, pick_threshold, svm_training_sample

# Same root-anchoring trick as clean_data.py and download_data.py.
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Data" / "final_data" / "f1_all_features.csv"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'TyreLife', 'TireDegDelta', 'Position', 'RaceProgress', 'LapsRemaining',
    'StopsSoFar', 'AnyCautionThisLap', 'CautionJustStarted', 'SafetyCarThisLap',
    'VSCThisLap', 'YellowThisLap', 'WetConditions', 'Stint', 'PittedThisLap',
]

CAUTION_FEATURES = ['AnyCautionThisLap', 'CautionJustStarted',
                    'SafetyCarThisLap', 'VSCThisLap', 'YellowThisLap']


def load_data():
    # 2025 is now a full season in the raw data, but the project's train/
    # validate/test split was built around 2018-2024, so we keep the cutoff.
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df['Year'] <= 2024]

    # A handful of early races (mostly 2018) are missing TyreLife or Stint on a
    # few hundred laps - an older-data gap in FastF1, not something we caused.
    # We drop them rather than guess values.
    before = len(df)
    df = df.dropna(subset=FEATURES)
    print("Dropped {} rows with missing features ({:.2f}% of data)".format(
        before - len(df), 100 * (before - len(df)) / before))

    # SPLIT: by YEAR - train 2018-2022, validate 2023, test 2024. Laps within
    # a race are heavily correlated, so a random split would leak.
    train = df[df['Year'] <= 2022]
    validate = df[df['Year'] == 2023]
    test = df[df['Year'] == 2024]

    print("\nTrain    2018-2022 | {:6d} laps | {:4d} pits ({:.2f}%)".format(
        len(train), train['PitNextLap'].sum(), 100 * train['PitNextLap'].mean()))
    print("Validate 2023      | {:6d} laps | {:4d} pits ({:.2f}%)".format(
        len(validate), validate['PitNextLap'].sum(), 100 * validate['PitNextLap'].mean()))
    print("Test     2024      | {:6d} laps | {:4d} pits ({:.2f}%)".format(
        len(test), test['PitNextLap'].sum(), 100 * test['PitNextLap'].mean()))

    return train, validate, test


def build_models():
    return {
        'Logistic Regression': Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(class_weight='balanced', max_iter=1000,
                                         random_state=RANDOM_STATE))]),

        'Gaussian Naive Bayes': Pipeline([
            ('scale', StandardScaler()),
            ('model', GaussianNB())]),

        'SVM (RBF kernel)': Pipeline([
            ('scale', StandardScaler()),
            ('model', SVC(class_weight='balanced', probability=True,
                          random_state=RANDOM_STATE))]),

        'Random Forest': RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1),

        'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }


def fit_model(name, model, fit_X, fit_y):
    if name == 'AdaBoost':
        ratio = (fit_y == 0).sum() / (fit_y == 1).sum()
        weights = np.where(fit_y == 1, ratio, 1.0)
        model.fit(fit_X, fit_y, sample_weight=weights)
    else:
        model.fit(fit_X, fit_y)
    return model


def run_naive_baseline(X_test, y_test):
    print("\nNAIVE BASELINE - pit when TyreLife > k")
    print("{:>4} {:>8} {:>10} {:>8}".format("k", "F1", "Precision", "Recall"))
    for k in [15, 20, 25, 30]:
        guess = (X_test['TyreLife'] > k).astype(int)
        print("{:>4} {:>8.4f} {:>10.4f} {:>8.4f}".format(
            k,
            f1_score(y_test, guess),
            precision_score(y_test, guess, zero_division=0),
            recall_score(y_test, guess)))


def run_all_models(train, validate, test):
    X_train, y_train = train[FEATURES], train['PitNextLap']
    X_val,   y_val   = validate[FEATURES], validate['PitNextLap']
    X_test,  y_test  = test[FEATURES], test['PitNextLap']

    run_naive_baseline(X_test, y_test)

    svm_train = svm_training_sample(train, 'PitNextLap')
    print("\nSVM trains on a {}-row stratified sample instead of the full {}-row "
          "training set (runtime, not accuracy, is the constraint here)".format(
              len(svm_train), len(train)))

    models = build_models()

    print("\nMODEL COMPARISON")
    print("{:22} {:>8} {:>7} {:>8} {:>10} {:>8}".format(
        "Model", "ROC AUC", "thresh", "F1", "Precision", "Recall"))

    results = {}
    for name, model in models.items():
        if name == 'SVM (RBF kernel)':
            fit_X, fit_y = svm_train[FEATURES], svm_train['PitNextLap']
        else:
            fit_X, fit_y = X_train, y_train

        model = fit_model(name, model, fit_X, fit_y)

        threshold = pick_threshold(y_val, model.predict_proba(X_val)[:, 1])
        probability = model.predict_proba(X_test)[:, 1]
        prediction = (probability >= threshold).astype(int)

        results[name] = {
            'auc':         roc_auc_score(y_test, probability),
            'threshold':   threshold,
            'f1':          f1_score(y_test, prediction),
            'precision':   precision_score(y_test, prediction, zero_division=0),
            'recall':      recall_score(y_test, prediction),
            'model':       model,
            'probability': probability,
        }
        r = results[name]
        print("{:22} {:>8.4f} {:>7.2f} {:>8.4f} {:>10.4f} {:>8.4f}".format(
            name, r['auc'], r['threshold'], r['f1'], r['precision'], r['recall']))

    best_name = max(results, key=lambda n: results[n]['auc'])
    print("\nBest by ROC AUC: {} ({:.4f})".format(best_name, results[best_name]['auc']))

    return results, X_test, y_test


def run_diagnostics(results, X_test, y_test, test):
    print("\nDIAGNOSTIC - single-feature AUC vs Random Forest importance")
    forest = results['Random Forest']['model']
    importance = dict(zip(FEATURES, forest.feature_importances_))

    print("{:22} {:>10} {:>12}".format("Feature", "Solo AUC", "RF import."))
    rows = [(f, roc_auc_score(y_test, X_test[f]), importance[f]) for f in FEATURES]
    for feature, auc, imp in sorted(rows, key=lambda r: -r[2]):
        print("{:22} {:>10.4f} {:>12.4f}".format(feature, auc, imp))

    print("\nPit rate by race phase (test year, 2024):")
    bins = [0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.01]
    labels = ['0-15%', '15-30%', '30-45%', '45-60%', '60-75%', '75-100%']
    phase = pd.cut(test['RaceProgress'], bins=bins, labels=labels, right=False)
    for p in labels:
        subset = test[phase == p]
        print("  {:8s} n={:5d}   {:5.2f}%".format(p, len(subset), 100 * subset['PitNextLap'].mean()))


def run_ablation(train, test, y_train, y_test, forest_full, probability_full):
    print("\nABLATION - Random Forest with vs without caution features")
    reduced = [f for f in FEATURES if f not in CAUTION_FEATURES]

    forest_reduced = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=20, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1)
    forest_reduced.fit(train[reduced], y_train)

    probability_reduced = forest_reduced.predict_proba(test[reduced])[:, 1]

    auc_full    = roc_auc_score(y_test, probability_full)
    auc_reduced = roc_auc_score(y_test, probability_reduced)

    print("  with caution features:    {:.4f}".format(auc_full))
    print("  without caution features: {:.4f}".format(auc_reduced))
    print("  difference:               {:+.4f}".format(auc_full - auc_reduced))

    caution_laps = (test['AnyCautionThisLap'] == 1).values
    auc_full_caution = auc_reduced_caution = None
    if caution_laps.sum() > 0 and y_test[caution_laps].nunique() > 1:
        auc_full_caution = roc_auc_score(y_test[caution_laps], probability_full[caution_laps])
        auc_reduced_caution = roc_auc_score(y_test[caution_laps], probability_reduced[caution_laps])
        print("\n  On caution laps only (n={}):".format(caution_laps.sum()))
        print("    with:    {:.4f}".format(auc_full_caution))
        print("    without: {:.4f}".format(auc_reduced_caution))

    return {
        'auc_full': auc_full, 'auc_reduced': auc_reduced,
        'auc_full_caution': auc_full_caution, 'auc_reduced_caution': auc_reduced_caution,
        'n_caution': caution_laps.sum(), 'n_all': len(y_test),
    }


# FIGURE 1: Model comparison bar chart

def make_model_comparison_figure(results):
    names = list(results.keys())
    aucs = [results[n]['auc'] for n in names]
    colors = ['#4C72B0'] * len(names)
    best_idx = int(np.argmax(aucs))
    colors[best_idx] = '#DD8452'

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, aucs, color=colors)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random guessing (0.50)')
    ax.set_ylabel('ROC AUC (test year: 2024)')
    ax.set_title('Model comparison — predicting a pit stop one lap ahead')
    ax.set_ylim(0.4, 0.9)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, auc + 0.01, '{:.3f}'.format(auc),
               ha='center', fontsize=9)
    plt.xticks(rotation=15, ha='right', fontsize=8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_model_comparison.png", dpi=150)
    plt.close()
    print("Saved {}".format(OUTPUT_DIR / "fig_model_comparison.png"))


# FIGURE 2: Feature importance (Random Forest)

def make_feature_importance_figure(forest):
    importance = pd.Series(forest.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance.index, importance.values, color='#4C72B0')
    ax.set_xlabel('Random Forest feature importance')
    ax.set_title('Which features the model actually relies on')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_feature_importance.png", dpi=150)
    plt.close()
    print("Saved {}".format(OUTPUT_DIR / "fig_feature_importance.png"))


# FIGURE 3: Ablation - with vs without caution features

def make_ablation_figure(ablation):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    width = 0.35
    full_vals = [ablation['auc_full'], ablation['auc_full_caution']]
    reduced_vals = [ablation['auc_reduced'], ablation['auc_reduced_caution']]

    bars1 = ax.bar(x - width/2, full_vals, width, label='With caution features', color='#DD8452')
    bars2 = ax.bar(x + width/2, reduced_vals, width, label='Without caution features', color='#4C72B0')
    ax.set_xticks(x)
    ax.set_xticklabels(['All test laps\n(n={:,})'.format(ablation['n_all']),
                        'Caution laps only\n(n={:,})'.format(ablation['n_caution'])])
    ax.set_ylabel('ROC AUC')
    ax.set_title('Ablation: does the caution feature set earn its place?')
    ax.set_ylim(0.5, 0.9)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, '{:.3f}'.format(h), ha='center', fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_ablation.png", dpi=150)
    plt.close()
    print("Saved {}".format(OUTPUT_DIR / "fig_ablation.png"))


def main():
    train, validate, test = load_data()
    y_train, y_test = train['PitNextLap'], test['PitNextLap']

    results, X_test, y_test = run_all_models(train, validate, test)
    run_diagnostics(results, X_test, y_test, test)

    forest = results['Random Forest']['model']
    probability_full = results['Random Forest']['probability']
    ablation = run_ablation(train, test, y_train, y_test, forest, probability_full)

    make_model_comparison_figure(results)
    make_feature_importance_figure(forest)
    make_ablation_figure(ablation)

    print("\nAll supervised figures saved to {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()