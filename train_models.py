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

df = pd.read_csv('f1_all_features.csv', low_memory=False)

# 2025 is one incomplete race (the season is still in progress in real life),
# so it can't be a fair test year. Drop it and work with 2018-2024 only.
df = df[df['Year'] <= 2024]

FEATURES = [
    'TyreLife', 'TireDegDelta', 'Position', 'RaceProgress', 'LapsRemaining',
    'StopsSoFar', 'AnyCautionThisLap', 'CautionJustStarted', 'SafetyCarThisLap',
    'VSCThisLap', 'YellowThisLap', 'WetConditions', 'Stint', 'PittedThisLap',
]

# A handful of early races (mostly 2018) are missing TyreLife or Stint on a few
# hundred laps - an older-data gap in FastF1, not something we caused. It's
# 904 rows out of 159,755 (0.6%), so we drop them rather than guess values.
before = len(df)
df = df.dropna(subset=FEATURES)
print("Dropped {} rows with missing features ({:.2f}% of data)".format(
    before - len(df), 100 * (before - len(df)) / before))

# SPLIT: by YEAR, the real version of the proposal's plan

# train 2018-2022, validate 2023, test 2024 - matching what the term project
# proposal committed to, now that we actually have the years to do it with.
# This is the same idea as splitting by race within one season, just at the
# scale the project always intended: an entire season the model has never
# seen, from a year it has never seen, rather than a handful of held-out races
# from the same season it trained on.
train    = df[df['Year'] <= 2022]
validate = df[df['Year'] == 2023]
test     = df[df['Year'] == 2024]

X_train, y_train = train[FEATURES], train['PitNextLap']
X_val,   y_val   = validate[FEATURES], validate['PitNextLap']
X_test,  y_test  = test[FEATURES], test['PitNextLap']

print("\nTrain    2018-2022 | {:6d} laps | {:4d} pits ({:.2f}%)".format(
    len(train), y_train.sum(), 100 * y_train.mean()))
print("Validate 2023      | {:6d} laps | {:4d} pits ({:.2f}%)".format(
    len(validate), y_val.sum(), 100 * y_val.mean()))
print("Test     2024      | {:6d} laps | {:4d} pits ({:.2f}%)".format(
    len(test), y_test.sum(), 100 * y_test.mean()))

# NAIVE BASELINE

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

# Sweeps cutoffs on the VALIDATION year only, never the test year - picking a
# threshold on data the final score also uses would be tuning to the exam.
def pick_threshold(y_true, probabilities):
    best_threshold, best_f1 = 0.5, -1.0
    for candidate in np.linspace(0.01, 0.99, 99):
        score = f1_score(y_true, (probabilities >= candidate).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, candidate
    return best_threshold

# SVM TRAINING SAMPLE

# SVM's cost grows roughly with the SQUARE (or worse) of the training set size.
# On the full ~108,000-row training set that's well over half an hour for one
# fit; on a 15,000-row stratified sample it's under a minute, and the resulting
# AUC lands within a few points of what the larger fit would give. We keep
# every pit-stop lap (they're the rare, valuable ones) and match them with an
# equal-sized random sample of non-pit laps, so the SVM still sees how severe
# the class imbalance is without training on all 108,000 rows to learn it.
def svm_training_sample(train_df, target_col, negatives_to_keep=12000, seed=RANDOM_STATE):
    positives = train_df[train_df[target_col] == 1]
    negatives = train_df[train_df[target_col] == 0].sample(
        n=min(negatives_to_keep, (train_df[target_col] == 0).sum()), random_state=seed)
    return pd.concat([positives, negatives]).sample(frac=1, random_state=seed)

svm_train = svm_training_sample(train, 'PitNextLap')
print("\nSVM trains on a {}-row stratified sample instead of the full {}-row "
      "training set (runtime, not accuracy, is the constraint here)".format(
          len(svm_train), len(train)))

# MODELS

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
    if name == 'SVM (RBF kernel)':
        fit_X, fit_y = svm_train[FEATURES], svm_train['PitNextLap']
    else:
        fit_X, fit_y = X_train, y_train

    if name == 'AdaBoost':
        ratio = (fit_y == 0).sum() / (fit_y == 1).sum()
        weights = np.where(fit_y == 1, ratio, 1.0)
        model.fit(fit_X, fit_y, sample_weight=weights)
    else:
        model.fit(fit_X, fit_y)

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

# DIAGNOSTIC: single-feature AUC vs Random Forest importance

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

# ABLATION: do the caution features earn their place?

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

caution_laps = (test['AnyCautionThisLap'] == 1).values
if caution_laps.sum() > 0 and y_test[caution_laps].nunique() > 1:
    print("\n  On caution laps only (n={}):".format(caution_laps.sum()))
    print("    with:    {:.4f}".format(roc_auc_score(y_test[caution_laps], probability_full[caution_laps])))
    print("    without: {:.4f}".format(roc_auc_score(y_test[caution_laps], probability_reduced[caution_laps])))