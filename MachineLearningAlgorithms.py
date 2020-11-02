"""
Does machine learning things.

Original author: Daniel Cohen, Summer 2019 Competitive Intelligence Analyst Intern
"""


import datetime as dt  # standard libraries
import time

import pandas as pd  # third-party libraries
import numpy as np
import tensorflow as tf; tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR); tf.set_random_seed(420)
import tensorflow.feature_column as fc
import seaborn as sns; sns.set_style("darkgrid")

from Utils import change_dir, max_day
from Plots import plot_from_date_indexed_df


PROB_THRESHOLD = 0.15


def split_5_folds(df):
    """
    Splits df into 5 manually-chosen time categories, where each contains 1 (or 2 very similar) recessions;
    this function exists because each recession is different and I don't want a model that is trained
    on any data point from a recession it is attempting to predict.
    :param df: DataFrame to split
    :return: df with 'group' column that splits into 5 groups (0 through 4)
    """

    def group_number(date):
        """
        Helper function that returns group number based on manually-chosen date ranges.
        :param date: date
        :return: group date falls into
        """

        date = pd.Timestamp(date)
        if date >= dt.datetime(year=1971, month=6, day=1) and date < dt.datetime(year=1977, month=6, day=1):
            return 0
        elif date >= dt.datetime(year=1976, month=6, day=1) and date < dt.datetime(year=1986, month=4, day=1):
            return 1
        elif date >= dt.datetime(year=1986, month=4, day=1) and date < dt.datetime(year=1996, month=11, day=1):
            return 2
        elif date >= dt.datetime(year=1996, month=11, day=1) and date < dt.datetime(year=2007, month=5, day=1):
            return 3
        else:
            return 4

    df['group'] = df['date'].map(group_number)
    return df


# def split_k_folds(df, k):
#     """
#     Takes DataFrame and splits into training set and test set.
#     :param df: DataFrame to split
#     :param k: Number of divisions to divide DataFrame into
#     :return: df with 'group' column that splits into k groups (0 through k - 1)
#     """
#
#     df_copy = df.copy(deep=True)
#     rows_per_group = (len(df) // k) + 1
#     df_copy.sample(frac=1)  # use if you want random sampling, comment out if you don't
#     df_copy['group'] = [i // rows_per_group for i in range(len(df))]
#     # df_train = df_copy[df_copy['group'] != div]
#     # df_test = df_copy[df_copy['group'] == div]
#     return df_copy


def plot_all_dep_vars_vs_preds(df_augmented, dep_vars):
    """
    Takes augmented DataFrame from augment_with_multivar_bootstrapped_boosted_tree_regression(), and, for each
    dependent variable in dep_vars, creates a separate plot of *dep_var* and *dep_var*_predicted vs. date
    :param df_augmented: DataFrame returned by augment_with_multivar_bootstrapped_boosted_tree_regression(), with
    *col*_predicted for each col in dep_vars
    :param dep_vars: list of dependent variables
    :return: nothing, but exports one graph to Figures per dependent variable
    """

    for var in dep_vars:
        pred_col = "{}_predicted".format(var)
        # accuracy = len(df_augmented[df_augmented[var] == df_augmented[pred_col]]) / len(df_augmented)
        plot_from_date_indexed_df(df_augmented,
                                  [var, pred_col],
                                  {var: var, pred_col: pred_col},
                                  var + " - " + pred_col,
                                  normalize=False,
                                  cutoff_lines=True)


def three_month_predictions(df, models_list, relevant_cols, num_offset_days):
    """
    Makes recessions probability predictions for the next three months.
    :param df: DataFrame returned by augment_with_multivar_bootstrapped_boosted_tree_regression()
    :param models_list: list of models returned by five_fold_multivar_bstrapped_boosted_tree()
    :param relevant_cols: columns used in creation of model
    :param num_offset_days: number of days in advance the dependent var is predicted; divide by 31 for months
    :return: [month_1_pred, month_2_pred, month_3_pred], assuming current month is month_0
    """

    df_copy = df.copy(deep=True)
    try:
        df_copy = df_copy[relevant_cols]
    except KeyError:
        print("Columns not the same between model and test data. Returning NAs.")
        return [None, None, None]

    def prev_month_and_year(month, year):
        """
        Returns previous month and year to month, year.
        :param month: month
        :param year: year
        :return: prev_month, prev_year
        """

        return (month - 1, year) if month - 1 > 0 else (12, year - 1)

    def x_month_pred(x):
        """
        Makes prediction x months in advance; for use only within three_month_prediction
        :param x: number of months in advance to predict
        :return: value between 0 and 1 of predicted recession probability, and its standard deviation
        """

        print("Making prediction for {} months in advance.".format(x))
        curr_year = dt.datetime.now().year
        curr_month = dt.datetime.now().month

        preds = []
        for i in range(1, int((num_offset_days / 31) + 1)):  # for each month in prediction timeframe
            curr_month, curr_year = prev_month_and_year(curr_month, curr_year)
            if i >= 3:  # exclude 1-, 2-, 3-month predictions b/c insufficient data
                if x + i <= 6:
                    relevant_model_name = 'USREC{}month'.format(x + i)
                    print("Predicting with {}.".format(relevant_model_name))
                    spec_data = df_copy[df_copy['date'] >= dt.datetime(year=curr_year, month=curr_month, day=1)]
                    spec_data = spec_data[spec_data['date'] <= dt.datetime(year=curr_year, month=curr_month,
                                                                           day=max_day(curr_month))]
                    spec_data = spec_data.reset_index(drop=True).drop(['date'], 1)

                    # get list of all the relevant models (i.e. get all 5 USREC1month models)
                    relevant_models = [models_list[i][relevant_model_name]['model'] for i in range(len(models_list))]
                    for est in relevant_models:
                        test_input_fn = make_input_fn(spec_data, [], len(spec_data), shuffle=False, n_epochs=1)
                        pred_dicts = list(est.predict(test_input_fn))
                        probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
                        preds.extend(probs.values)
                    print("Current average probability:", np.mean(preds))
        # print('mean probability:', np.mean(preds))
        # print('std probability:', np.std(preds))
        return {'prob': np.mean(preds), 'std': np.std(preds)}

    prediction_dicts = [x_month_pred(1), x_month_pred(2), x_month_pred(3)]
    preds = [pred['prob'] * 100 for pred in prediction_dicts]  # convert from proportion to percentage

    change_dir("Table exports")
    preds_table = pd.DataFrame(columns=['preds'], data=preds)
    preds_table.to_csv("curr_preds.csv", index=False)

    return preds


def augment_with_multivar_bootstrapped_boosted_tree_regression(five_versions, df, dep_vars):
    """
    Augments df with "predictions" for each dep_var in dep_vars, performing num_reps-x boosted tree regression
    for each dependent variable; for each value of k, the predicted values for group k are the values from an
    algorithm that was trained using all data not in group k.
    :param five_versions: list of boostrapped multivariate models returned by five_fold_multivar_bstrapped_boosted_tree
    :param df: DataFrame to split into df_train and df_test (already prepped for training); MUST BE SAME DATAFRAME
           AS WAS USED FOR CREATING ALGORITHMS!
    :param dep_vars: list of dependent variables
    :return: df, with one additional column (*dep_var*_predicted) for each dep_var
    """

    k = 5
    df_list = []
    for i in range(k):  # create df with predictive dep_var columns for the date range of each particular k
        df_group_k = df[df['group'] == i]
        dep_var_model_dict = five_versions[i]
        for var in dep_vars:  # create predictive column for each dep_var within that date range
            spec_model = dep_var_model_dict[var]
            pred_col = "{}_predicted".format(var)
            curr_probs = spec_model['probs']
            temp_probs_df = pd.DataFrame(data=curr_probs.values, columns=[pred_col], index=curr_probs.index)
            df_group_k = pd.merge(df_group_k, temp_probs_df, left_index=True, right_index=True, how='outer')
        df_list.append(df_group_k)
    df_merged = pd.concat(df_list)  # concat them all together --> df with full timeframe and pred col for each dep var
    return df_merged


def five_fold_multivar_bstrapped_boosted_tree(df, indep_vars, dep_vars, num_reps):
    """
    Executes multivar_bootstrapped_boosted_tree_regression() for each k.
    :param df: DataFrame to split into df_train and df_test (doesn't yet have group column)
    :param indep_vars: list of keys of independent variables
    :param dep_vars: list of dependent variables
    :param num_reps: number of times to bootstrap model
    :return: list with 5 elements, each element being: dict that maps dep_var to dict of
             {'model': best_model, 'probs': best_probs, 'accuracy': best_accuracy}
    """

    model_dicts = []
    df_split = split_5_folds(df)
    for k in range(5):
        df_train, df_test = df_split[df_split['group'] != k], df_split[df_split['group'] == k]
        model_dicts.append(
            multivar_bootstrapped_boosted_tree_regression(df_train, df_test, indep_vars, dep_vars, num_reps, k))
    return model_dicts


def multivar_bootstrapped_boosted_tree_regression(df_train, df_test, indep_vars, dep_vars, num_reps, k):
    """
    For each dependent variable: performs boosted tree regression num_reps times on df_train, tests on df_test, and
    returns estimator with highest accuracy.
    :param df_train: DataFrame containing date column/index and independent-variable/dependent-variable columns
    :param df_test: like above but test
    :param indep_vars: list of keys of independent variables
    :param dep_vars: list of dependent variables
    :param num_reps: number of times to bootstrap model
    :param k: current value of k, to be used for naming convention
    :return: dict mapping dep_var to: dict of {'model': best_model, 'probs': best_probs, 'accuracy': best_accuracy}
    """

    best_model_dict = {}
    for dep_var in dep_vars:
        best_accuracy = 0
        for i in range(num_reps):
            new = boosted_tree_regression(df_train, df_test, indep_vars, dep_var, k, i)
            if new is not None and new['accuracy'] > best_accuracy:
                best = new
        best_model_dict[dep_var] = best
    return best_model_dict


def make_input_fn(X, y, num_examples, n_epochs=None, shuffle=True):
    """
    Helper function to use with TensorFlow.
    :param X: independent-variable data in np.ndarray format
    :param y: dependent-variable data as one-dimensional np.ndarray
    :param num_examples: length of training set
    :param n_epochs: number of epochs
    :param shuffle: whether or not to shuffle the data
    :return: function that, when called, will return the relevant TF dataset
    """
    def input_fn():
        if len(y) > 0:
            dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        else:  # if you're predicting, there will be no y-values
            dataset = tf.data.Dataset.from_tensor_slices(dict(X))
        if shuffle:
            dataset = dataset.shuffle(num_examples)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(num_examples)
        return dataset

    return input_fn


def boosted_tree_regression(df_train, df_test, indep_vars, dep_var_col, k, rep):
    """
    Executes boosted tree classification on df with indep_vars vs. dep_var_col;
    Source: https://www.tensorflow.org/tutorials/estimators/boosted_trees
    :param df_train: DataFrame containing date column/index and independent-variable/dependent-variable columns
    :param df_test: like above but test
    :param indep_vars: list of keys of independent variables
    :param dep_var_col: name of dependent variable
    :param k: current value of k, to be used for naming convention
    :param rep: current iteration of models corresponding to k, to be used for naming convention
    :return: {'model': best_model, 'probs': best_probs, 'accuracy': best_accuracy}
    """

    try:
        # create file name
        file_name = "tf_boostedtreesestimator_k{}_rep{}".format(k, rep)

        # define independent columns, and produce training/test sets
        indep_cols = [col for col in df_train.columns.values if col in indep_vars]
        # df_split = split_k_folds(df, k=k)  # k-fold cross validation

        results = []
        print("Training new model on %s..." % dep_var_col)
        start_time = time.time()

        # produce training/test DFs
        # df_train = df_split[df_split['group'] != i]
        # df_test = df_split[df_split['group'] == i]

        df_train = df_train.reset_index(drop=True).drop(['date'], 1)
        df_test_index = df_test.index  # get index to set for probabilities Series
        df_test = df_test.reset_index(drop=True).drop(['date'], 1)
        # produce training X/Y, test X/Y
        # train_X = df_train.loc[:, indep_cols].values
        train_Y = df_train.pop(dep_var_col)
        # test_X = df_test.loc[:, indep_cols].values
        test_Y = df_test.pop(dep_var_col)

        # normalize x-values
        # train_X = min_max_normalized(train_X)
        # test_X = min_max_normalized(test_X)

        # set up model framework
        numeric_columns = indep_cols
        feature_columns = []
        for feature_name in numeric_columns:
            feature_columns.append(fc.numeric_column(feature_name, dtype=tf.float32))

        # define model
        num_examples = len(train_Y)

        # Training and evaluation input functions.
        train_input_fn = make_input_fn(df_train, train_Y, num_examples)
        # eval_input_fn = make_input_fn(df_test, test_Y, num_examples, shuffle=False, n_epochs=1)
        eval_input_fn = make_input_fn(df_test, [], num_examples, shuffle=False, n_epochs=1)

        # train model
        n_batches = 1
        est = tf.estimator.BoostedTreesClassifier(feature_columns,
                                                  n_batches_per_layer=n_batches)

        # The model will stop training once the specified number of trees is built, not
        # based on the number of steps.
        est.train(train_input_fn, max_steps=100)

        # Eval.
        # results = est.evaluate(eval_input_fn)
        # print('Accuracy : ', results['accuracy'])
        # print('Dummy model: ', results['accuracy_baseline'])

        pred_dicts = list(est.predict(eval_input_fn))
        probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])#.reindex(index=df_test_index)
        probs = pd.DataFrame(data=probs.values, columns=['probs'], index=df_test_index)['probs']
        predicted_vals = [1 if prob > PROB_THRESHOLD else 0 for prob in probs]
        accuracy = len(df_test[test_Y == predicted_vals]) / len(df_test)

        end_time = time.time()
        print("Successfully trained new model in %s seconds." % (end_time - start_time))

        return {'model': est, 'probs': probs, 'accuracy': accuracy}
    except KeyError as e:
        print("Error while training model:", e)
        return


# def log_reg_complete(df_test, log_reg_results, indep_vars, dep_var):
#     """
#     Takes test DataFrame and computes predictions for each row, using indep_vars; then computes accuracy of
#     predictions using df_test, and plots prediction and actual against date.
#     :param df_test: test DataFrame
#     :param log_reg_results: model that results from training on training/validation data
#     :param indep_vars: list of independent variables
#     :param dep_var: dependent variable
#     :return: Augmented version of df_test with 'prediction_*dep_var*' column
#     """
#
#     print("Computing predictions for %s." % dep_var)
#     start_time = time.time()
#
#     pred_col = 'prediction_{}'.format(dep_var)
#     df_test[pred_col] = [make_log_reg_prediction(log_reg_results, df_test.iloc[i], indep_vars)
#                          for i in range(len(df_test))]
#     accuracy = len(df_test[df_test[pred_col] == df_test[dep_var]]) / len(df_test)
#     print("Accuracy of logistic regression model: %s." % accuracy)
#
#     plot_from_date_indexed_df(df_test, [dep_var, pred_col], {dep_var: dep_var, pred_col: pred_col}, "accuracy",
#                               normalize=False)
#
#     end_time = time.time()
#     print("Successfully computed predictions for %s in %s seconds." % (dep_var, (end_time - start_time)))
#     return df_test
#
#
# def make_log_reg_prediction(log_reg_results, data, indep_vars):
#     """
#     Takes list of bootstrapped k-fold cross-validated logistic-regression [A, b] lists; applies data --> prediction.
#     :param log_reg_results: list of: num_reps lists of: k lists, each containing [A, b]
#     :param data: one row of data with the same columns as training/validation table
#     :param indep_vars: independent variables
#     :return: 1 or 0
#     """
#
#     cols = [col for col in data.index.values if col in indep_vars]
#     data = np.array(data[cols].values).reshape(1, len(cols)).astype(dtype='float32')
#     predictions = []
#     for l in log_reg_results:  # for each k-fold cross-validated set of (A, b) pairs
#         temp_predictions = []
#         for pair in l:  # for each [A, b] pair
#             A = pair[0]
#             b = pair[1]
#             pred = data@A + b
#             temp_predictions.append(1 if pred[0][0] > 0 else 0)  # @: built-in matrix multiplication for NumPy
#         try:
#             predictions.append(max(set(temp_predictions), key=temp_predictions.count))
#         except StatisticsError:
#             predictions.append(1)  # if they're even, predict a recession
#     try:
#         return max(set(predictions), key=predictions.count)
#     except StatisticsError:
#         return 1  # if they're even, predict a recession
#
#
# def bootstrapped_k_fold_logistic_regression(df, indep_vars, dep_var_col, k, num_reps):
#     """
#     Performs k-fold cross-validated logistic regression of indep_vars vs. dep_var_col in df.
#     :param df: DataFrame containing date column/index and independent-variable/dependent-variable columns
#     :param indep_vars: dict mapping keys of independent variables to names
#     :param dep_var_col: name of dependent variable column to be regressed against
#     :param k: value for k-fold cross-validation (suggested between 5 and 10)
#     :param num_reps: for bootstrapping; number of repetitions to do entire process
#     :return: list of: num_reps lists of: k lists, each containing [A, b]
#     """
#
#     print("Making %s trained models." % (num_reps * k))
#     return [k_fold_logistic_regression(df, indep_vars, dep_var_col, k) for i in range(num_reps)]
#
#
# def k_fold_logistic_regression(df, indep_vars, dep_var_col, k):
#     """
#     Executes logistic regression on df with indep_vars vs. dep_var_col;
#     Source: https://www.kaggle.com/autuanliuyc/logistic-regression-with-tensorflow
#     :param df: DataFrame containing date column/index and independent-variable/dependent-variable columns
#     :param indep_vars: list of keys of independent variables
#     :param dep_var_col: name of dependent variable column to be regressed against
#     :param k: value for k-fold cross-validation (suggested between 5 and 10)
#     :return: list of k [A, b] values
#     """
#
#     try:
#         # define independent columns, and produce training/test sets
#         indep_cols = [col for col in df.columns.values if col in indep_vars]
#         df_split = split_k_folds(df, k=k)  # k-fold cross validation
#
#         results = []
#         for i in range(k):  # k-fold cross validation
#             print("Training new model...")
#
#             # produce training/test DFs
#             df_train = df_split[df_split['group'] != i]
#             df_test = df_split[df_split['group'] == i]
#
#             # produce training X/Y, test X/Y
#             train_X = df_train.loc[:, indep_cols].values
#             train_Y = df_train.loc[:, dep_var_col].values
#             test_X = df_test.loc[:, indep_cols].values
#             test_Y = df_test.loc[:, dep_var_col].values
#
#             # normalize x-values
#             train_X = min_max_normalized(train_X)
#             test_X = min_max_normalized(test_X)
#
#             # set up model framework
#             A = tf.Variable(tf.random.normal(shape=[len(indep_cols), 1]))
#             b = tf.Variable(tf.random.normal(shape=[1, 1]))
#             init = tf.compat.v1.global_variables_initializer()
#             with tf.compat.v1.Session() as sess:
#                 sess.run(init)
#
#                 # print('A before', A.eval(session=sess))
#                 # print('b before:', b.eval(session=sess))
#
#                 data = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, len(indep_cols)])
#                 target = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1])
#                 mod = tf.matmul(data, A) + b
#                 loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mod, labels=target))
#
#                 learning_rate = 0.0035
#                 batch_size = 100
#                 iter_num = 1000
#
#                 opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
#                 goal = opt.minimize(loss)
#
#                 prediction = tf.round(tf.sigmoid(mod))
#                 correct = tf.cast(tf.equal(prediction, target), dtype=tf.float32)
#                 accuracy = tf.reduce_mean(correct)
#
#                 loss_trace = []
#                 train_acc = []
#                 test_acc = []
#
#                 for epoch in range(iter_num):
#                     # Generate random batch index
#                     batch_index = np.random.choice(len(train_X), size=batch_size)
#                     batch_train_X = train_X[batch_index]
#                     batch_train_Y = np.array(train_Y[batch_index]).reshape(1, batch_size).T
#
#                     sess.run(goal, feed_dict={data: batch_train_X, target: batch_train_Y})
#                     temp_loss = sess.run(loss, feed_dict={data: batch_train_X, target: batch_train_Y})
#                     # convert into a matrix, and the shape of the placeholder to correspond
#                     temp_train_acc = sess.run(accuracy, feed_dict={data: train_X,
#                                                                    target: np.array(train_Y).reshape(1, len(train_X)).T})
#                     temp_test_acc = sess.run(accuracy, feed_dict={data: test_X,
#                                                                   target: np.array(test_Y).reshape(1, len(test_X)).T})
#                     # recode the result
#                     loss_trace.append(temp_loss)
#                     train_acc.append(temp_train_acc)
#                     test_acc.append(temp_test_acc)
#                     # output
#                     # if (epoch + 1) % 300 == 0:
#                     #     print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss,
#                     #                                                                              temp_train_acc,
#                     #                                                                              temp_test_acc))
#
#                 # print('A after', A.eval())
#                 # print('b after:', b.eval())
#                 # plt.plot(train_acc, 'b-', label='train accuracy')
#                 # plt.plot(test_acc, 'k-', label='test accuracy')
#                 # plt.xlabel('epoch')
#                 # plt.ylabel('accuracy')
#                 # plt.title('Train and Test Accuracy')
#                 # plt.legend(loc='best')
#                 # change_dir("Figures")
#                 # plt.savefig("accuracy-test{}.png".format(seg), format='png', dpi=1000)
#
#                 results.append([A.eval(), b.eval()])
#         return results
#     except KeyError as e:
#         print("Error creating feature matrix/data labels:", e)
#         return


def sort_set_index_and_rename(df):
    """
    Takes DataFrame from Data Warehouse; sorts, sets 'date' column as index, renames index 'date_index'.
    :param df: DataFrame to modify
    """

    try:
        df.sort_values('date', inplace=True)
        df.set_index('date', drop=False, inplace=True)
        df.index.rename('date_index', inplace=True)
    except KeyError as e:
        print("Error in setting index by date:", e)


def prep_for_training(df, train_start, train_end):
    """
    Takes DataFrame like dc_model_merged or dc_model_fred_vware_combined; isolates data between training_start,
    training_end and returns as training/validation table with proper columns.
    :param df: DataFrame with date column and column for each independent/offset-dependent variable
    :param train_start: start-date for training set
    :param train_end: end-date for training set
    :return: DataFrame of training/validation data, list of columns in dataset
    """

    print("Preparing data for training and testing.")
    start_time = time.time()
    try:
        sort_set_index_and_rename(df)
        df_copy = df.loc[train_start <= df['date']].loc[df['date'] <= train_end].dropna(axis='columns', how='any')
        # print("Independent variables:", df_copy.columns.values)
        end_time = time.time()
        print("Successfully prepared data for training and testing in %s seconds." % (end_time - start_time))
        return df_copy, df_copy.columns.values
    except KeyError as e:
        print("Error while filtering DataFrame on date:", e)
        return df, df.columns.values


# def timeframe_threshold(df, thresh, indep_vars):
#     """
#     Takes in DataFrame like dc_model_merged or dc_model_fred_vware_combined and returns DataFrame
#     containing counts of variables and whether or not that date was above the threshold.
#     :param df: DataFrame with date column and column for each independent/offset-dependent variable
#     :param thresh: number of variables that must be non-null for given date to be "good"
#     :param indep_vars: dict mapping independent variable keys to names; used to make sure we don't count
#            non-null dependent variables.
#     :return: DataFrame with index date_index, columns date & count
#     """
#
#     df_copy = df.copy(deep=True)
#     try:
#         df_copy = df_copy[['date'] + [col for col in df_copy.columns.values if col in indep_vars.keys()]]
#         sort_set_index_and_rename(df_copy)
#     except KeyError as e:
#         print("Error converting 'date' column to index:", e)
#         raise SystemExit
#     df_copy['count'] = df_copy.count(axis=1).values - 1  # account for date column
#     df_copy['threshold_met'] = [1 if v >= thresh else 0 for v in df_copy['count']]
#     return df_copy[['date', 'count', 'threshold_met']]
