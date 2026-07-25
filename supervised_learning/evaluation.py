import pandas as pd
from sklearn.metrics import f1_score

RANDOM_STATE = 6740

# Sweeps cutoffs on the VALIDATION year only, never the test year - picking a
# threshold on data the final score also uses would be tuning to the exam.
def pick_threshold(y_true, probabilities):
    best_threshold, best_f1 = 0.5, -1.0
    for candidate in [i / 100 for i in range(1, 100)]:
        score = f1_score(y_true, (probabilities >= candidate).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, candidate
    return best_threshold

# SVM's cost grows roughly with the SQUARE (or worse) of the training set size.
# On the full training set that's well over half an hour for one fit; on a
# stratified sample it's under a minute, and the resulting AUC lands within a
# few points of what the larger fit would give. We keep every pit-stop lap
# (they're the rare, valuable ones) and match them with a random sample of
# non-pit laps, so the SVM still sees how severe the class imbalance is
# without training on the full training set to learn it.
def svm_training_sample(train_df, target_col, negatives_to_keep=12000, seed=RANDOM_STATE):
    positives = train_df[train_df[target_col] == 1]
    negatives = train_df[train_df[target_col] == 0].sample(
        n=min(negatives_to_keep, (train_df[target_col] == 0).sum()), random_state=seed)
    return pd.concat([positives, negatives]).sample(frac=1, random_state=seed)