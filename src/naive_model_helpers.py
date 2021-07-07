import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from patsy import dmatrices
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from IPython.display import display, Image


def train_test_split_by_ga(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("has_dental_ga", axis=1),
        df.has_dental_ga,
        test_size=0.33,
        stratify=df.has_dental_ga,
        random_state=1
    )
    df_train = X_train
    df_train["has_dental_ga"] = y_train

    df_test = X_test
    df_test["has_dental_ga"] = y_test

    return X_train, X_test


def patsify_data(df):

    if "entity_id" in df.columns:
        df.drop("entity_id", axis=1, inplace=True)

    features = df.drop("has_dental_ga", axis=1).columns
    patsy_string = "has_dental_ga ~ "
    patsy_string += " + ".join(features)
    y, X = dmatrices(patsy_string, data=df, return_type="dataframe")

    return y, X


def log_reg_model_of_ga(df):

    y, X = patsify_data(df)

    mod = sm.Logit(y, X)
    log_reg = mod.fit(maxiter=100)

    return log_reg


def plot_ga_rate_comparison(df, feature):

    df = df[[feature, "has_dental_ga"]].copy()
    df["count"] = np.ones(len(df))
    df["has_dental_ga"] = df.has_dental_ga.map({1:"yes", 0: "no"}).astype("category")
    feature_counts = (df.groupby([feature, "has_dental_ga"])
                      .agg("count").reset_index())
    for feature_val in df[feature].unique():
        val_map = feature_counts[feature] == feature_val
        counts = feature_counts[val_map]["count"].values
        sum = counts.sum()
        feature_counts.loc[val_map, "pct"] = counts * 100 / sum

    fig = make_subplots(rows=1, cols=2,
                        column_widths=[0.34, 0.66],
                        horizontal_spacing=0.17)

    has_ga = feature_counts.has_dental_ga == "yes"
    pct_chart = go.Bar(x=feature_counts[has_ga][feature],
                       y=feature_counts[has_ga]["pct"],
                       marker={"color": "#EF553B"})
    fig.add_trace(pct_chart, row=1, col=1)
    fig["data"][0]["showlegend"] = False

    has_ga_counts_bar = go.Bar(x=feature_counts[has_ga][feature],
                               y=feature_counts[has_ga]["count"],
                               name="Had Dental GA")
    fig.add_trace(has_ga_counts_bar, row=1, col=2)


    no_ga_counts_bar = go.Bar(x=feature_counts[~has_ga][feature],
                              y=feature_counts[~has_ga]["count"],
                              name="No Dental GA")

    fig.add_trace(no_ga_counts_bar, row=1, col=2)
    fig.update_xaxes(title=feature, tickangle=35)
    fig.update_yaxes(title="Percentage Had GA", row=1, col=1)
    fig.update_yaxes(title="Count", row=1, col=2)
    fig.update_layout(
        width=1000,
        height=500,
        title=dict(text=f"Percentage / Counts of GA experience by {feature}",
                   x=0.5)
    )
    fig_image = fig.to_image(format="png")
    display(Image(fig_image))

