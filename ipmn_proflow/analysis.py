from imports import *


def analysis_importance(best_model, columns_name):
    # TODO: show tree, separate importance, graph
    # TODO: show/not by config
    # 获取特征重要性
    feature_importances = best_model.feature_importances_
    # 创建 DataFrame，使特征名称与重要性对应
    importance_df = pd.DataFrame({
        'Feature': columns_name,  # 使用预存的列名
        'Importance': feature_importances
    })
    # 按重要性排序
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    # 可视化特征重要性
    plt.figure(figsize=(12, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.xlabel("Importance Score")
    plt.ylabel("Feature Name")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()  # 让重要性最高的特征在顶部
    plt.show()
    # 输出特征重要性数据
    print(importance_df)


def analysis_performance(config, y_test, test_probabilities):
    global y_pred

    # Evaluate the model using ROC-AUC score
    test_auc = roc_auc_score(y_test, test_probabilities)
    print("Test AUC: ", test_auc)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, test_probabilities)

    # show curve
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")

    # set tpr for evaluate
    if config.TPR_SET == 0:
        # use preset TPR
        desired_tpr = config.TPR
        closest_threshold = thresholds[np.argmin(np.abs(tpr - desired_tpr))]
        y_pred = (test_probabilities >= closest_threshold).astype(int)
        print(f"Using preset TPR: {desired_tpr}, Closest Threshold: {closest_threshold}")

    elif config.TPR_SET == 1:
        # use mouse click position
        def onclick(event):
            global y_pred
            if event.xdata is not None and event.ydata is not None:
                desired_tpr = event.ydata  # user mouse click choice
                closest_threshold = thresholds[np.argmin(np.abs(tpr - desired_tpr))]

                # predict laundering in 0/1 bool, based on closest_threshold
                y_pred = (test_probabilities >= closest_threshold).astype(int)
                print(f"Selected TPR: {desired_tpr}, Closest Threshold: {closest_threshold}")

        # monitor click
        fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    cm = confusion_matrix(y_test, y_pred)

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
    print(classification_report(y_test, y_pred))
