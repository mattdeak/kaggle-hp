import tensorflow as tf
import numpy as np

class F1Hook(tf.train.SessionRunHook):

    def begin(self):
        # Add tensor to graph for summaries before finalization
        print ("Adding F1 Tracker to Graph")
        self._F1Tensor = tf.placeholder(tf.float32, [])
        self.summary_op = tf.summary.scalar("F1-Global", self._F1Tensor)

    def after_create_session(self, session, coord):

        tp_name = 'metrics/f1_macro/TP'
        fp_name = 'metrics/f1_macro/FP'
        fn_name = 'metrics/f1_macro/FN'
        
        tp_tensor = session.graph.get_tensor_by_name(tp_name+':0')
        fp_tensor = session.graph.get_tensor_by_name(fp_name+':0')
        fn_tensor = session.graph.get_tensor_by_name(fn_name+':0')
        
        
        self.tp = np.zeros(28)
        self.fp = np.zeros(28)
        self.fn = np.zeros(28)
        
        self.args = [tp_tensor, fp_tensor, fn_tensor]
        print(f"Got Args: {self.args}")
        
    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self.args)
    
    def after_run(self, run_context, run_values):
        tp, fp, fn = run_values.results
        
        self.tp = np.sum([self.tp, tp], axis=0)
        self.fp = np.sum([self.fp, fp], axis=0)
        self.fn = np.sum([self.fn, fn], axis=0)
        
    def end(self, session):
        print("Calculating metrics")
        print(self.tp.shape, self.fp.shape)
        precision = self.tp / np.sum([self.tp, self.fp], axis=0)
        recall = self.tp / np.sum([self.tp, self.fn], axis=0)
        
        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)
        
        f1 = np.nan_to_num(2*(precision * recall)/(precision + recall))

        print (f'Global F-1: {np.mean(f1)}')
        # Write to tensorboard
        session.run(self.summary_op, feed_dict={self._F1Tensor: np.mean(f1)})
        