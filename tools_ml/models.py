from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer



def get_pipeline(num_features: list, cat_features: list, estimator):
    pipe_tr_num_features = Pipeline([
        ('tr_imput_mean', SimpleImputer(strategy='mean')),
        ('tr_min_max', StandardScaler())
    ])

    pipe_tr_cat_features = Pipeline([
        ('tr_input_frequent', SimpleImputer(strategy='most_frequent')),
        ('tr_dummy', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    pre_processor = ColumnTransformer([
        ('tr_num', pipe_tr_num_features, num_features),
        ('tr_cat', pipe_tr_cat_features, cat_features)
    ])

    final_pipe = Pipeline([
        ('pre_processor', pre_processor),
        ('estimator', estimator)
    ])

    return final_pipe