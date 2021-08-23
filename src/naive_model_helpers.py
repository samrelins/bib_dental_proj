import numpy as np
import pandas as pd
from patsy import dmatrices
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
import statsmodels.api as sm
from IPython.display import display, Image


def train_test_split_by_target(df, target):
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target, axis=1),
        df[target],
        test_size=0.33,
        stratify=df[target],
        random_state=1
    )
    df_train = X_train
    df_train[target] = y_train

    df_test = X_test
    df_test[target] = y_test

    return X_train, X_test


def patsify_data(df, feature):

    if "entity_id" in df.columns:
        df.drop("entity_id", axis=1, inplace=True)

    features = df.drop(feature, axis=1).columns
    patsy_string = f"{feature} ~ "
    patsy_string += " + ".join(features)
    y, X = dmatrices(patsy_string, data=df, return_type="dataframe")

    return y, X


def log_reg_model_of_feature(df, feature, disp=True):

    y, X = patsify_data(df, feature)

    mod = sm.Logit(y, X)
    results = mod.fit(maxiter=100,
                      disp=disp)

    return results


def plot_target_rate_comparison(df, feature, target, plot_title=None,
                                percentages=True):

    fig = make_subplots(rows=1, cols=2,
                        column_widths=[0.34, 0.66],
                        vertical_spacing=0.2,
                        horizontal_spacing=0.17)

    raw_counts = df[feature].value_counts()
    feature_counts_bar = go.Bar(x=raw_counts.index,
                               y=raw_counts.values)
    fig.add_trace(feature_counts_bar, row=1, col=1)
    fig["data"][0]["showlegend"] = False

    df = df[[feature, target]].copy()
    df["count"] = np.ones(len(df))
    feature_counts = (df.groupby([feature, target])
                      .agg("count").reset_index())
    for has_ga in [1, 0]:

        ga_map = feature_counts[target] == has_ga
        counts = feature_counts[ga_map]["count"].values
        sum = counts.sum()
        feature_counts.loc[ga_map, "pct"] = counts * 100 / sum

        outcome = target + " = True" if has_ga == 1 else target + " = False"
        name = f"% of {outcome}"

        if percentages:
            pct_chart = go.Bar(x=feature_counts[ga_map][feature],
                               y=feature_counts[ga_map]["pct"],
                               name=name)
        else:
            pct_chart = go.Bar(x=feature_counts[ga_map][feature],
                               y=feature_counts[ga_map]["count"],
                               name=name)

        fig.add_trace(pct_chart, row=1, col=2)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=1.15,
        xanchor="right",
        x=1.1
    ))


    target_name = target.replace("_", " ").title()
    feature_name = feature.replace("_", " ").title()
    if plot_title is None:
        plot_title = (f"{target_name} experience by {feature_name}")
    fig.update_xaxes(tickangle=35)
    fig.update_yaxes(title="Count", row=1, col=1)
    if percentages:
        fig.update_yaxes(title="Percent", row=1, col=2)
    else:
        fig.update_yaxes(title="Count", row=1, col=2)
    fig.update_layout(
        width=2400,
        height=1400,
        title=dict(text=plot_title,
                   x=0.5),
        font=dict(size=30)
    )
    fig_image = fig.to_image(format="png")
    display(Image(fig_image))


def cv_train_gb_classifier(df, target):
    param_grid = {"n_estimators": [10, 30, 100, 130],
                  "max_depth": [2, 4, 10, 14],
                  "min_samples_split": [100, 300, 1000, 1300],
                  "max_features": [5, 10, 15, 20, 25]}

    gb_clf = GradientBoostingClassifier()
    cv_clf = GridSearchCV(gb_clf,
                          param_grid,
                          scoring="roc_auc",
                          n_jobs=14,
                          verbose=1)

    y_train, X_train = patsify_data(df, target)
    cv_clf.fit(X_train,
               y_train[target])

    return cv_clf


# function to return LR/GB model performance stats
def return_model_performance_stats(model, train_df, test_df):
    is_gb_model = type(model) == GridSearchCV
    if is_gb_model:
        _, X_train = patsify_data(train_df, "caries_experience")
        train_preds = model.predict_proba(X_train)[:,1]
        _, X_test = patsify_data(test_df, "caries_experience")
        test_preds = model.predict_proba(X_test)[:,1]
    else:
        train_preds = model.predict(train_df)
        test_preds = model.predict(test_df)

    train_roc_auc = roc_auc_score(train_df.caries_experience,
                                  train_preds)
    test_roc_auc = roc_auc_score(test_df.caries_experience,
                                 test_preds)

    print(f"Train ROC/AUC = {train_roc_auc}. Test ROC/AUC = {test_roc_auc}")

    if is_gb_model:
        r_squared = np.nan
    else:
        r_squared = model.prsquared

    train_fpr, train_tpr, _ = roc_curve(train_df.caries_experience,
                                        train_preds)
    train_roc_curve = (train_fpr, train_tpr)
    test_fpr, test_tpr, _ = roc_curve(test_df.caries_experience,
                                      test_preds)
    test_roc_curve = (test_fpr, test_tpr)
    return r_squared, train_roc_auc, train_roc_curve, test_roc_auc, test_roc_curve


def cramers_v(feature_x, feature_y, dataf):
    conf_matrix = pd.crosstab(dataf[feature_x].values,
                              dataf[feature_y].values)
    chi2 = stats.chi2_contingency(conf_matrix)[0]
    n = np.sum(conf_matrix.values)
    phi2 = chi2 / n
    r,k = conf_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


def build_cramersv_df(df, cat_cols):
    results = None
    for col_x in cat_cols:
        col_x_results = []
        for col_y in cat_cols:
            col_x_y_cv = cramers_v(col_x, col_y, df)
            col_x_results.append(col_x_y_cv)
        col_x_results = pd.DataFrame(col_x_results,
                                     index=cat_cols,
                                     columns=[col_x])
        if results is None:
            results = col_x_results
        else:
            results = results.join(col_x_results)
    return results


def plot_roc_curves(plot_data, title=None):

    fig = go.Figure()
    for name, data in plot_data.items():
        fpr, tpr = data
        fig.add_trace(
            go.Scatter(x=fpr,
                       y=tpr,
                       name=name)
        )
    fig.update_layout(
        title=dict(text=title,
                   x=0.5),
        width=800,
        height=600,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )
    fig_image = fig.to_image(format="png")
    display(Image(fig_image))
