
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import collections
from sklearn.base import clone
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

def print_class_distribution(label_file, label_name="Label", output_filename=None):
    labels = np.load(label_file)
    counter = collections.Counter(labels)
    total = len(labels)
    
    output_lines = [f"📊 Class Distribution for: {label_name} ({total} samples)", "-" * 40]
    for cls, count in sorted(counter.items()):
        percentage = 100 * count / total
        output_lines.append(f"Class {cls}: {count} samples ({percentage:.2f}%)")
    
    output_text = "\n".join(output_lines)
    print(output_text)
    
    if output_filename:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(output_text)

def evaluation(
    model,
    features_path: str,
    labels_path: str,
    label_name: str = "Label",
    cv_strategy=None,
    groups_path: str = None,
    multi_class="ovr"
):
    X = np.load(features_path)
    y = np.load(labels_path)
    groups = np.load(groups_path) if groups_path else None
    
    if cv_strategy is None:
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    is_logo = isinstance(cv_strategy, LeaveOneGroupOut)
    average_mode = "micro" if is_logo else "macro"
    
    accs, f1s, precs, recalls, aucs, names = [], [], [], [], [], []
    all_y_true, all_y_pred, all_y_proba = [], [], []
    
    print("Features Extracted from: ", features_path)
    print(f"\n CV {cv_strategy.__class__.__name__} Evaluation for: {label_name} using {model.__class__.__name__}")
    print("-" * 90)
    print(f"{'Fold/Group':<15}{'Acc':>8}{'F1':>8}{'Prec':>10}{'Recall':>10}{'AUROC':>10}")
    print("-" * 90)
    
    for i, (train_idx, val_idx) in enumerate(cv_strategy.split(X, y, groups=groups), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        group_val = groups[val_idx[0]] if groups is not None else i
        
        # Calculate class weights if supported
        supports_weight = "class_weight" in model.get_params().keys()
        clf_params = model.get_params()
        if supports_weight and clf_params.get("class_weight", None) is None:
            try:
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
                weight_dict = {cls: w for cls, w in zip(np.unique(y_train), class_weights)}
                model.set_params(class_weight=weight_dict)
            except Exception as e:
                print(f"Warning: Could not set class_weight: {e}")
        
        clf = clone(model)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        
        try:
            y_proba = clf.predict_proba(X_val)
        except:
            y_proba = None
        
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average=average_mode, zero_division=0)
        prec = precision_score(y_val, y_pred, average=average_mode, zero_division=0)
        rec = recall_score(y_val, y_pred, average=average_mode, zero_division=0)
        
        accs.append(acc)
        f1s.append(f1)
        precs.append(prec)
        recalls.append(rec)
        names.append(group_val)
        
        auc = float('nan')
        if y_proba is not None:
            try:
                if multi_class == "ovr" and y_proba.shape[1] > 2:
                    auc = roc_auc_score(y_val, y_proba, multi_class=multi_class, average="macro")
                else:
                    # Binary case
                    if len(y_proba.shape) == 1:
                        auc = roc_auc_score(y_val, y_proba)
                    else:
                        auc = roc_auc_score(y_val, y_proba[:, 1])
            except Exception:
                pass
        aucs.append(auc)
        
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        if y_proba is not None:
            if len(y_proba.shape) == 2 and y_proba.shape[1] > 1:
                all_y_proba.extend(y_proba[:, 1])
            else:
                all_y_proba.extend(y_proba)
        
        print(f"{str(group_val):<15}{acc:.3f}{f1:>8.3f}{prec:>10.3f}{rec:>10.3f}{auc:>10.3f}")
        
        # Show CM only for first few folds or all if small number
        if not is_logo or (i <= 3):
            # We don't want to block execution with plt.show(), so we might just print or save
            # For now, let's skip the blocking plot
            pass
        
    print("\n📈 Average:")
    print(f"  Accuracy  ={np.mean(accs):.3f}")
    print(f"  Precision ={np.mean(precs):.3f}")
    print(f"  Recall    ={np.mean(recalls):.3f}")
    print(f"  {average_mode.title()}-F1 ={np.mean(f1s):.3f}")
    print(f"  AUROC     ={np.nanmean(aucs):.3f}")
    
    # Global LOGO metrics
    if is_logo:
        print("\n✅ Global Evaluation (LOGO):")
        global_acc = accuracy_score(all_y_true, all_y_pred)
        global_f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
        global_rec = recall_score(all_y_true, all_y_pred, average="macro", zero_division=0)
        global_prec = precision_score(all_y_true, all_y_pred, average="macro", zero_division=0)
        
        try:
             global_auc = roc_auc_score(all_y_true, all_y_proba) if len(all_y_proba) == len(all_y_true) else float('nan')
        except:
             global_auc = float('nan')
        
        print(f"  Accuracy  ={global_acc:.3f}")
        print(f"  Precision ={global_prec:.3f}")
        print(f"  Recall    ={global_rec:.3f}")
        print(f"  Macro-F1  ={global_f1:.3f}")
        print(f"  AUROC     ={global_auc:.3f}")
        
    result_df = pd.DataFrame({
        "Fold_or_Group": names,
        "Accuracy": accs,
        "Precision": precs,
        "Recall": recalls,
        f"{average_mode.title()}-F1": f1s,
        "AUROC": aucs
    })
    
    return result_df
