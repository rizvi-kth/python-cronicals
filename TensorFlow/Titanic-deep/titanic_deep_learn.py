from titanic_Data_Prep import get_tf_feature_columns
import tensorflow as tf

# import sys
# for path in sys.path:
#     print(path)

my_feature_columns = get_tf_feature_columns()
print("Feature columns: {}".format(len(my_feature_columns)))

for col in my_feature_columns:
    print(type(col))

# Build 2 hidden layer DNN with 10, 10 units respectively.
# Configure to log in a directory
# Configure to log every 50 epoch for Console Output. Make sure to INFO log by tf.logging.set_verbosity(tf.logging.INFO)
# Configure to log every 50 epoch for Tensorboard.
classifier = tf.estimator.Estimator(model_fn=my_model,
                                    model_dir='./iris-nn-impl/log1',
                                    config=tf.estimator.RunConfig(log_step_count_steps=50, save_summary_steps = 50),
                                    params={
                                        'feature_columns': my_feature_columns,
                                        # Two hidden layers of 10 nodes each.
                                        'hidden_units': [16, 16],
                                        # The model must choose between 3 classes.
                                        'n_classes': 3,
                                    })
