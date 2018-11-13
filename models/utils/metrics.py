import tensorflow as tf 
import tensorflow.keras.backend as K
from lib.config import NUM_CLASSES

def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator

def _f1_macro_vector(y_true, y_pred):
    """Computes the F1-score with Macro averaging.
    
    Arguments:
        y_true {tf.Tensor} -- Ground-truth labels
        y_pred {tf.Tensor} -- Predicted labels
    
    Returns:
        tf.Tensor -- The computed F1-Score
    """
    y_true = K.cast(y_true, tf.float64)
    y_pred = K.cast(y_pred, tf.float64)
    
    TP = tf.reduce_sum(y_true * K.round(y_pred), axis=0, name='TP')
    FN = tf.reduce_sum(y_true * (1 - K.round(y_pred)), axis=0, name='FN')
    FP = tf.reduce_sum((1 - y_true) * K.round(y_pred), axis=0, name='FP')
    
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    
    # Convert NaNs to Zero
    prec = tf.where(tf.is_nan(prec), tf.zeros_like(prec), prec)
    rec = tf.where(tf.is_nan(rec), tf.zeros_like(rec), rec)
    
    f1 = 2 * (prec * rec) / (prec + rec)
    
    # Convert NaN to Zero
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return f1


def f1_macro(y_true, y_pred):
    """Computes the F1-score with Macro averaging.
    
    Arguments:
        y_true {tf.Tensor} -- Ground-truth labels
        y_pred {tf.Tensor} -- Predicted labels
    
    Returns:
        tf.Tensor -- The computed F1-Score
    """
    f1 = _f1_macro_vector(y_true, y_pred)
    
    # tf.merge(precision_summaries)
    # tf.merge(recall_summaries)
        
#     tf.summary.scalar('Class_Precisions', prec)
#     tf.summary.scalar('Class_Recalls', rec)

    f1_mean = K.mean(f1)
    
    return f1_mean

def make_class_specific_f1(label_number):

    @rename(f'Class{label_number}_F1-Macro')
    def label_f1_macro(y_true, y_pred):
        f1 = _f1_macro_vector(y_true, y_pred)
        return f1[label_number]

    return label_f1_macro