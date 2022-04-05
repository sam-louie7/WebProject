

def add_delta_columns(df, col, new_col):
    df[new_col] = df[col] - df[col].shift(fill_value=0)
    return df


def groupby_with_func(df, col1, col2 , func):
    X = df.groupby(col1).agg({col2: func})
    return X.to_numpy()


def filter_and_groupby_with_func(df, col1, col2 , func, val):
    df[col1] = df[df[col1] >= val]
    X = df.groupby(col1).agg({col2: func})
    return X.to_numpy()


