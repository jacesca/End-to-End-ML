import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from environment import (SEED, hprint, print, save_object,
                         FEATURES_SELECTED_PATH, SCALER_PATH,
                         SAVED_FIGURE_PATH)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


def feature_engineering(df: pd.DataFrame,
                        target_col: str,
                        show_graph: bool = True) \
                            -> tuple[pd.DataFrame, pd.DataFrame,
                                     pd.Series, pd.Series, StandardScaler]:
    """
    Output:
        X_train, X_test, y_train, y_test, standardizer
    """
    hprint('Feature engineering')

    # Split into X and y
    X = df.drop(target_col, axis='columns')
    y = df[target_col]

    # Split training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=SEED)

    # Standardizing the data
    standardizer = StandardScaler()
    X_train_std = pd.DataFrame(data=standardizer.fit_transform(X_train),
                               columns=X_train.columns)
    X_test_std = pd.DataFrame(data=standardizer.transform(X_test),
                              columns=X_test.columns)

    # Saving the scaler
    save_object(standardizer, SCALER_PATH)

    if show_graph:
        # Observing the transformation in one column
        col = 'age'
        legend_colors = [('Original', 'khaki'), ('Standardized', 'darkkhaki')]
        fig = plt.figure()
        plt.suptitle('Feature Engineering: Standardizing')
        for i, feature in enumerate([X_train[col], X_train_std[col]]):
            plt.subplot(1, 2, i+1)
            plt.title(f'{legend_colors[i][0]} "{col.capitalize()}" column')
            plt.hist(feature, bins=30, alpha=0.5, color=legend_colors[i][1],
                     label=legend_colors[i][0])
            plt.xlabel(col.capitalize())
            plt.ylabel('Count')
            plt.tight_layout()
        fig.savefig(f'{SAVED_FIGURE_PATH}feature_eng_standardizing.png')

    # Feature selection
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced',
                                max_depth=5)
    selector = SelectFromModel(rf)
    X_train_featured = selector.fit_transform(X_train_std, y_train)
    features_bool = selector.get_support()
    selected_features = X_train_std.columns[features_bool]

    # Saving the selected features
    save_object(selected_features, FEATURES_SELECTED_PATH)

    X_train_featured = pd.DataFrame(
        data=X_train_featured,
        columns=selected_features
    )
    X_test_featured = pd.DataFrame(
        data=selector.transform(X_test_std),
        columns=selected_features
    )

    print('Original features:', list(X_train.columns))
    print('Selected features:', list(selected_features))

    if show_graph:
        feature_importance = pd.DataFrame({
            'features': selector.feature_names_in_,
            'importance': selector.estimator_.feature_importances_, # or coef_ >> estimator dependency, current estimator RandomForestClassifier  # noqa
            'class': list(map(lambda x: 'Selected' if x else 'Discarded',
                              features_bool))
        })
        fig = plt.figure()
        sns.barplot(data=feature_importance,
                    x='importance', y='features',
                    hue='class',
                    palette=['red', 'green'])
        plt.title('Feature Engineering: Feature selection')
        fig.savefig(f'{SAVED_FIGURE_PATH}feature_selection.png')

    return (X_train_featured, X_test_featured, y_train, y_test,
            standardizer, selector)

    # # Another way to do it
    # rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced',
    #                             max_depth=5)
    # rf.fit(X_train_std, y_train)
    # selector = SelectFromModel(rf, prefit=True)
    # features_bool = selector.get_support()
    # features =  rf.feature_names_in_,
    # importance = rf.feature_importances_[features_bool]
