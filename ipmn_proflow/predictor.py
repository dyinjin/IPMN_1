from imports import *
from parameter_handler import *


config = load_config()

dataset = '2023-07.csv'
# dataset = 'sampled_IBM.csv'
model_path = f"{config.DATAPATH}20250626_050343_saved_model.pkl"
transformer_path = f"{config.DATAPATH}20250626_050343_saved_transformer.pkl"
config.SAVE_LEVEL = 0

data_set = UnitDataLoader.csvloader_specified(config, dataset)
data_set = UnitDataLoader.datauniter_saml(config, data_set)
# data_set = UnitDataLoader.csvloader_specified(config, config.IBM_CSV)
# data_set = UnitDataLoader.datauniter_ibm(config, data_set)

data_set = data_set.reset_index(drop=True)

y_pred_ori = data_set[config.STANDARD_INPUT_LABEL]
X_pred = data_set.drop(columns=config.STANDARD_INPUT_LABEL)

# Handle parameters based on mode specified in arguments
print(f"Parameter handle by mode: {config.PARAMETER_MODES}")
X_pred = parameter_adder(config, config.PARAMETER_MODES, X_pred)

columns_name = X_pred.columns
# already divide in week day hour etc.
X_pred = X_pred.drop(columns=config.STANDARD_TIME_PARAM)

loaded_transformer = joblib.load(transformer_path)
print("Transformer loaded successfully!")

X_pred = loaded_transformer.transform(X_pred)

# load saved model
loaded_model = joblib.load(model_path)
print("Model loaded successfully!")

pred_probabilities = loaded_model.predict_proba(X_pred)[:, 1]
pred_auc = roc_auc_score(y_pred_ori, pred_probabilities)
print("Predict AUC: ", pred_auc)

if config.SAVE_LEVEL == 0:
    pass
elif config.SAVE_LEVEL == 1:
    pd.DataFrame(pred_probabilities).to_csv(f"{config.DATAPATH}{dataset}-{param}-y_prob.csv", index=False)
elif config.SAVE_LEVEL == 2:
    pred_probabilities = pd.Series(pred_probabilities, name="predict_fraud_probability")
    pd.concat([pd.DataFrame(X_pred, columns=columns_name), y_pred_ori, pred_probabilities], axis=1).to_csv(
        f"{config.DATAPATH}{dataset}-{param}-X_test_with_prob.csv", index=False)

fpr, tpr, thresholds = roc_curve(y_pred_ori, pred_probabilities)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc="lower right")

if config.TPR_SET == 0:
    desired_tpr = config.TPR
    closest_threshold = thresholds[np.argmin(np.abs(tpr - desired_tpr))]
    y_pred = (pred_probabilities >= closest_threshold).astype(int)
    print(f"Using preset TPR: {desired_tpr}, Closest Threshold: {closest_threshold}")

elif config.TPR_SET == 1:
    def onclick(event):
        global y_pred
        if event.xdata is not None and event.ydata is not None:
            desired_tpr = event.ydata
            closest_threshold = thresholds[np.argmin(np.abs(tpr - desired_tpr))]

            y_pred = (pred_probabilities >= closest_threshold).astype(int)
            print(f"Selected TPR: {desired_tpr}, Closest Threshold: {closest_threshold}")


    fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

cm = confusion_matrix(y_pred_ori, y_pred)

# Plot confusion matrix heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title(f'Confusion Matrix')
plt.show()

# Extract confusion matrix metrics
tn, fp, fn, tp = cm.ravel()
fpr_cm = fp / (fp + tn)
tpr_cm = tp / (tp + fn)

# Print evaluation metrics
print(f"False Positive Rate (FPR): {fpr_cm:.3f}")
print(f"True Positive Rate (TPR): {tpr_cm:.3f}")

# Print classification report for detailed model evaluation
print(classification_report(y_pred_ori, y_pred))

if config.SAVE_LEVEL == 0:
    pass
elif config.SAVE_LEVEL == 1:
    pd.DataFrame(y_pred).to_csv(f"{config.DATAPATH}{dataset}-{param}-y_pred.csv", index=False)
elif config.SAVE_LEVEL == 2:
    y_pred = pd.Series(y_pred, name="predict_fraud")
    pd.concat([pd.DataFrame(X_pred, columns=columns_name), y_pred_ori, y_pred], axis=1).to_csv(
        f"{config.DATAPATH}{dataset}-{param}-X_test_with_pred.csv", index=False)
