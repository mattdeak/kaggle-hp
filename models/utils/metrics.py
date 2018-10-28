import tensorflow as tf 
import tensorflow.keras.backend as K

def f1_macro(y_true, y_pred):
    """Computes the F1-score with Macro averaging.
    
    Arguments:
        y_true {tf.Tensor} -- Ground-truth labels
        y_pred {tf.Tensor} -- Predicted labels
    
    Returns:
        tf.Tensor -- The computed F1-Score
    """
    y_true = K.cast(y_true, tf.float64)
    y_pred = K.cast(y_pred, tf.float64)
    
    TP = K.sum(y_true * K.round(y_pred), axis=0)
    FN = K.sum(y_true * (1 - K.round(y_pred)), axis=0)
    FP = K.sum((1 - y_true) * K.round(y_pred), axis=0)
    
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    
    # Convert NaNs to Zero
    prec = tf.where(tf.is_nan(prec), tf.zeros_like(prec), prec)
    rec = tf.where(tf.is_nan(rec), tf.zeros_like(rec), rec)
    
    f1 = 2 * (prec * rec) / (prec + rec)
    
    # Convert NaN to Zero
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    f1 = K.mean(f1)
    
    return f1