import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

ANNOTATIONS_DIR = '/Users/jbonaventura/Desktop/Annotations'

# --- Load all feature CSVs and tag each row with its slide ID ---
csv_paths = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, '*_features.csv')))
frames = []
for path in csv_paths:
    slide_id = os.path.basename(path).replace('_features.csv', '')
    df = pd.read_csv(path)
    df['slide_id'] = slide_id
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

print(f'Loaded {len(csv_paths)} slides, {len(data)} total tiles')
print(data.groupby(['slide_id', 'label']).size().to_string())

# --- Define feature columns ---
# Everything except tile metadata and the label itself
METADATA_COLS = {'tile_id', 'origin_row', 'origin_col', 'czi_r_min', 'czi_c_min',
                 'czi_r_max', 'czi_c_max', 'label', 'slide_id'}
FEATURE_COLS = [c for c in data.columns if c not in METADATA_COLS]

print(f'\nUsing {len(FEATURE_COLS)} features: {FEATURE_COLS}')

# --- Leave-one-slide-out cross-validation ---
# Each slide takes a turn as the held-out test set.
# The RF trains on the other 5 slides and predicts on the held-out one.
# This tests whether the model generalises across slides, not just across tiles
# from the same slide (which would be too easy).
slide_ids = sorted(data['slide_id'].unique())
classes = sorted(data['label'].unique())

print('\n--- Leave-one-slide-out results ---')
for test_slide in slide_ids:
    train_data = data[data['slide_id'] != test_slide]
    test_data  = data[data['slide_id'] == test_slide]

    X_train = train_data[FEATURE_COLS].values
    y_train = train_data['label'].values
    X_test  = test_data[FEATURE_COLS].values
    y_test  = test_data['label'].values

    # class_weight='balanced' compensates for Muscle being ~6x more common than Immune
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(f'\nTest slide: {test_slide}  ({len(test_data)} tiles)')
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues')
    plt.title(f'Test slide: {test_slide}')
    plt.tight_layout()
    plt.show()

# --- Random stratified split (upper bound) ---
# Tiles from all slides are pooled and split randomly, keeping class proportions equal.
# Train/test sets will contain tiles from the same slides, so this is optimistic —
# but it tells you the ceiling performance if slide variance weren't a factor.
print('\n--- Random stratified split (80/20, averaged over 5 runs) ---')
X = data[FEATURE_COLS].values
y = data['label'].values

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
reports = []
for train_idx, test_idx in sss.split(X, y):
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X[train_idx], y[train_idx])
    y_pred = rf.predict(X[test_idx])
    report = classification_report(y[test_idx], y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    reports.append(pd.DataFrame(report).T)

# Average metrics across the 5 runs
avg_report = pd.concat(reports).groupby(level=0).mean()
print(avg_report.round(2).to_string())

cm = confusion_matrix(y[test_idx], y_pred, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap='Blues')
plt.title('Random stratified split (last fold)')
plt.tight_layout()
plt.show()

# --- Summary comparison ---
print('\n--- Macro F1 summary ---')
print('Leave-one-slide-out results above (see per-slide reports)')
print(f'Random stratified split macro F1: {avg_report.loc["macro avg", "f1-score"]:.2f}  (averaged over 5 runs)')
