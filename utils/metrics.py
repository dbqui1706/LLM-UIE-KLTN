def recall(y_true, y_pred):
    y_true = list(set(y_true))
    y_pred = list(set(y_pred))

    return len([x for x in y_pred if x in y_true]) / len(y_true)
