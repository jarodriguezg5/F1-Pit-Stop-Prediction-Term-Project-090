import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

RANDOM_STATE = 6740

df = pd.read_csv('f1_2024_features.csv')

FEATURES = [
    'TyreLife', 'TireDegDelta', 'Position', 'RaceProgress', 'LapsRemaining',
    'StopsSoFar', 'AnyCautionThisLap', 'CautionJustStarted', 'SafetyCarThisLap',
    'VSCThisLap', 'YellowThisLap', 'WetConditions', 'Stint', 'PittedThisLap',
]

# SPLIT: three-way, by race

# Splitting on whole races instead of random rows. Laps inside one race share a
# safety car period, weather, and track, so a random split would put lap 12 of
# Singapore in training and lap 13 in testing - the model would be graded on
# races it had effectively already seen.
#
# Three groups, not two, because we need somewhere to pick the decision threshold
# that isn't the test set:
#   train    rounds 1-14   -> fit the models
#   validate rounds 15-18  -> choose each model's threshold
#   test     rounds 19-24  -> final scoring, touched once
# This mirrors the 2018-2022 / 2023 / 2024-2025 year split we'll use at full scale.
train    = df[df['RoundNumber'] <= 14]
validate = df[(df['RoundNumber'] >= 15) & (df['RoundNumber'] <= 18)]
test     = df[df['RoundNumber'] >= 19]

X_train, y_train = train[FEATURES], train['PitNextLap']
X_val,   y_val   = validate[FEATURES], validate['PitNextLap']
X_test,  y_test  = test[FEATURES], test['PitNextLap']

print("Train    rounds 1-14  | {:5d} laps | {:3d} pits ({:.2f}%)".format(
    len(train), y_train.sum(), 100 * y_train.mean()))
print("Validate rounds 15-18 | {:5d} laps | {:3d} pits ({:.2f}%)".format(
    len(validate), y_val.sum(), 100 * y_val.mean()))
print("Test     rounds 19-24 | {:5d} laps | {:3d} pits ({:.2f}%)".format(
    len(test), y_test.sum(), 100 * y_test.mean()))

# NAIVE BASELINE

# The dumb rule any fan could state without machine learning. Everything we build
# has to beat it, or the ML wasn't worth doing.
print("\nNAIVE BASELINE - pit when TyreLife > k")
print("{:>4} {:>8} {:>10} {:>8}".format("k", "F1", "Precision", "Recall"))
for k in [15, 20, 25, 30]:
    guess = (X_test['TyreLife'] > k).astype(int)
    print("{:>4} {:>8.4f} {:>10.4f} {:>8.4f}".format(
        k,
        f1_score(y_test, guess),
        precision_score(y_test, guess, zero_division=0),
        recall_score(y_test, guess)))

# THRESHOLD PICKER

# Every model outputs a probability, not a yes/no. Turning that into a decision
# needs a cutoff, and 0.5 is a terrible default when only 3% of laps are positive -
# AdaBoost and SVM almost never cross it, which is why they scored F1 = 0.
#
# So we sweep cutoffs and keep whichever maximises F1. Critically this runs on the
# VALIDATION races, never the test races: picking a threshold on test data would
# be tuning to the exam we're about to grade ourselves on.
def pick_threshold(y_true, probabilities):
    best_threshold, best_f1 = 0.5, -1.0
    for candidate in np.linspace(0.01, 0.99, 99):
        score = f1_score(y_true, (probabilities >= candidate).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, candidate
    return best_threshold

# MODELS

# StandardScaler puts every feature on the same scale (mean 0, sd 1). Coefficient-
# and distance-based models - logistic regression, naive bayes, SVM - get distorted
# when one feature runs 0-70 (TyreLife) and another runs 0-1 (flags). Trees split
# one feature at a time so scale is irrelevant to them; no scaler needed there.
#
# class_weight='balanced' tells a model the rare class matters. Without it,
# predicting "never pits" scores 97% accuracy and there's no reason to learn.
models = {
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

print("\nMODEL COMPARISON")
print("{:22} {:>8} {:>7} {:>8} {:>10} {:>8}".format(
    "Model", "ROC AUC", "thresh", "F1", "Precision", "Recall"))

results = {}
for name, model in models.items():
    # AdaBoost has no class_weight parameter, so we hand it sample weights
    # directly - each pit lap counts as much as the whole no-pit pile is heavy.
    if name == 'AdaBoost':
        ratio = (y_train == 0).sum() / (y_train == 1).sum()
        weights = np.where(y_train == 1, ratio, 1.0)
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train)

    threshold = pick_threshold(y_val, model.predict_proba(X_val)[:, 1])

    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(int)

    results[name] = {
        'auc':       roc_auc_score(y_test, probability),
        'threshold': threshold,
        'f1':        f1_score(y_test, prediction),
        'precision': precision_score(y_test, prediction, zero_division=0),
        'recall':    recall_score(y_test, prediction),
        'model':     model,
    }
    r = results[name]
    print("{:22} {:>8.4f} {:>7.2f} {:>8.4f} {:>10.4f} {:>8.4f}".format(
        name, r['auc'], r['threshold'], r['f1'], r['precision'], r['recall']))

best_name = max(results, key=lambda n: results[n]['auc'])
print("\nBest by ROC AUC: {} ({:.4f})".format(best_name, results[best_name]['auc']))

# DIAGNOSTIC: why the linear models struggle

# Solo AUC asks: if this were the ONLY thing we knew, how well could we rank laps
# by pit risk? 0.5 = useless, 1.0 = perfect. Comparing that against how heavily
# the forest leans on each feature exposes the non-linear relationships - a
# feature can look worthless alone yet be indispensable to a tree.
print("\nDIAGNOSTIC - single-feature AUC vs Random Forest importance")
forest = results['Random Forest']['model']
importance = dict(zip(FEATURES, forest.feature_importances_))

print("{:22} {:>10} {:>12}".format("Feature", "Solo AUC", "RF import."))
rows = [(f, roc_auc_score(y_test, X_test[f]), importance[f]) for f in FEATURES]
for feature, auc, imp in sorted(rows, key=lambda r: -r[2]):
    print("{:22} {:>10.4f} {:>12.4f}".format(feature, auc, imp))

# RaceProgress is the smoking gun: near-random on its own, yet the forest's top
# feature. Pit rate climbs through the middle of a race then collapses at the end,
# and a straight line can't be both increasing and decreasing. Logistic regression
# extracts nothing from it; a tree just carves the range into chunks.
print("\nPit rate by race phase (the shape linear models can't fit):")
bins = [0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.01]
labels = ['0-15%', '15-30%', '30-45%', '45-60%', '60-75%', '75-100%']
phase = pd.cut(test['RaceProgress'], bins=bins, labels=labels, right=False)
for p in labels:
    subset = test[phase == p]
    print("  {:8s} n={:5d}   {:5.2f}%".format(p, len(subset), 100 * subset['PitNextLap'].mean()))

# ABLATION: do the caution features earn their place?

# The experiment the proposal promised. Same model, same split, caution features
# stripped out - the AUC gap is what they contribute.
print("\nABLATION - Random Forest with vs without caution features")
caution_features = ['AnyCautionThisLap', 'CautionJustStarted',
                    'SafetyCarThisLap', 'VSCThisLap', 'YellowThisLap']
reduced = [f for f in FEATURES if f not in caution_features]

forest_reduced = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=20, class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1)
forest_reduced.fit(train[reduced], y_train)

probability_full    = forest.predict_proba(X_test)[:, 1]
probability_reduced = forest_reduced.predict_proba(test[reduced])[:, 1]

auc_full    = roc_auc_score(y_test, probability_full)
auc_reduced = roc_auc_score(y_test, probability_reduced)

print("  with caution features:    {:.4f}".format(auc_full))
print("  without caution features: {:.4f}".format(auc_reduced))
print("  difference:               {:+.4f}".format(auc_full - auc_reduced))

# Cautions fire on under 5% of laps, so their effect on a season-wide average is
# diluted. Scoring only the caution laps shows what the features are worth in the
# situation they were built for.
caution_laps = (test['AnyCautionThisLap'] == 1).values
if caution_laps.sum() > 0 and y_test[caution_laps].nunique() > 1:
    print("\n  On caution laps only (n={}):".format(caution_laps.sum()))
    print("    with:    {:.4f}".format(roc_auc_score(y_test[caution_laps], probability_full[caution_laps])))
    print("    without: {:.4f}".format(roc_auc_score(y_test[caution_laps], probability_reduced[caution_laps])))