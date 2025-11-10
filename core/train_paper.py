import pickle
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn Imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression  # Added from paper
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -----------------
# PLOT FUNCTIONS
# -----------------

def plot_feature_distribution(X_train, cs_f_features, X_train_enhanced_log, save_dir):
    """
    Figure 2: Plot distribution of a skewed feature before and after
    [cite_start]log transform to show the effect of the paper's method [cite: 420-422].
    """
    if not cs_f_features:
        print("No highly skewed features found to plot.")
        return

    # Pick the first skewed feature to visualize
    feature_to_plot = cs_f_features[0]

    plt.figure(figsize=(12, 5))

    # Plot original distribution
    plt.subplot(1, 2, 1)
    sns.histplot(X_train[feature_to_plot], kde=True, bins=50)
    plt.title(f'Original Distribution (Skew: {X_train[feature_to_plot].skew():.2f})')
    plt.xlabel(feature_to_plot)

    # Plot log-transformed distribution
    plt.subplot(1, 2, 2)
    sns.histplot(X_train_enhanced_log[feature_to_plot], kde=True, bins=50)
    plt.title(f'Log-Transformed Distribution (Skew: {X_train_enhanced_log[feature_to_plot].skew():.2f})')
    plt.xlabel(f'log({feature_to_plot})')

    plt.suptitle("Figure 2: Feature Distribution After Pre-processing")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figure_2_feature_distribution.png"))
    plt.close()

def plot_model_r2_comparison(scores_df, save_dir):
    """
    Figure 3: Plot R² scores for all models across all datasets.
    """
    plt.figure(figsize=(15, 7))
    sns.barplot(
        data=scores_df[scores_df['Metric'] == 'R2'],
        x='Model',
        y='Score',
        hue='Dataset'
    )
    plt.title("Figure 3: Model Performance Comparison (R² Scores)")
    plt.ylabel("R² Score")
    plt.xlabel("Model")
    plt.legend(title="Dataset", loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figure_3_model_r2_comparison.png"))
    plt.close()

def plot_prediction_vs_actual_grid(models_dict, X_test, y_test, save_dir):
    """
    Figure 4: Plot Prediction vs. Actual values in a grid for all models.
    """
    model_names = list(models_dict.keys())
    num_models = len(model_names)

    # Adjust grid size as needed
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten() # Flatten to 1D array for easy iteration

    if num_models > len(axes):
        print(f"Warning: More models ({num_models}) than plot axes ({len(axes)}). Some models will not be plotted.")
        model_names = model_names[:len(axes)]

    for i, name in enumerate(model_names):
        model = models_dict[name]
        y_pred = model.predict(X_test)

        ax = axes[i]
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(name)

    # Hide any unused subplots
    for j in range(num_models, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Figure 4: Model Predictions vs. Actual Values (Base Dataset)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(os.path.join(save_dir, "figure_4_prediction_vs_actual_grid.png"))
    plt.close()

def plot_random_forest_importance(model, features, save_dir, top_n=20):
    """
    Figure 5: Plot Random Forest feature importance.
    """
    importances = model.feature_importances_

    # Ensure top_n is not greater than the number of features
    top_n = min(top_n, len(features))

    indices = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(10, 8))
    plt.title(f"Figure 5: Top {top_n} Feature Importances (Random Forest)")
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figure_5_rf_feature_importance.png"))
    plt.close()

def plot_xgboost_convergence(model, save_dir):
    """
    Figure 6: Plot XGBoost training convergence (RMSE).
    """
    results = model.evals_result()
    plt.figure(figsize=(10, 6))
    plt.plot(results['validation_0']['rmse'], label='Train RMSE')
    plt.plot(results['validation_1']['rmse'], label='Validation RMSE')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title("Figure 6: XGBoost Training Convergence")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figure_6_xgb_convergence.png"))
    plt.close()

def plot_mlp_loss(model, save_dir):
    """
    Figure 7: Plot MLP architecture and training loss.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Figure 7: MLP Training Loss Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figure_7_mlp_training_loss.png"))
    plt.close()

def plot_all_metric_comparison(scores_df, save_dir):
    """
    Figure 8: Plot a grouped bar chart for all models and metrics
    on the Base dataset.
    """
    base_scores = scores_df[scores_df['Dataset'] == 'Base']

    plt.figure(figsize=(15, 8))
    sns.barplot(
        data=base_scores,
        x='Model',
        y='Score',
        hue='Metric'
    )
    plt.title("Figure 8: Model Comparison Across All Metrics (Base Dataset)")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figure_8_all_metric_comparison.png"))
    plt.close()

def plot_residuals_grid(models_dict, X_test, y_test, save_dir):
    """
    Figure 9: Plot Residuals vs. Predicted values in a grid for all models.
    """
    model_names = list(models_dict.keys())
    num_models = len(model_names)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    if num_models > len(axes):
        print(f"Warning: More models ({num_models}) than plot axes ({len(axes)}). Some models will not be plotted.")
        model_names = model_names[:len(axes)]

    for i, name in enumerate(model_names):
        model = models_dict[name]
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        ax = axes[i]
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals (Actual - Predicted)")
        ax.set_title(name)

    # Hide any unused subplots
    for j in range(num_models, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Figure 9: Residual Plots (Base Dataset)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(os.path.join(save_dir, "figure_9_residuals_grid.png"))
    plt.close()

def plot_svr_heatmap(grid_search, save_dir):
    """
    Figure 10: Plot a heatmap of SVR hyperparameter tuning results.
    This assumes 'C' and 'gamma' were in the param_grid.
    """
    if 'param_C' not in grid_search.cv_results_ or 'param_gamma' not in grid_search.cv_results_:
        print("Skipping SVR heatmap: 'C' or 'gamma' not in param_grid.")
        return

    try:
        # Pivot the results
        cv_results = pd.DataFrame(grid_search.cv_results_)
        pivot_table = cv_results.pivot_table(
            values='mean_test_score',
            index='param_C',
            columns='param_gamma'
        )

        plt.figure(figsize=(10, 7))
        sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis")
        plt.title("Figure 10: SVR Hyperparameter Heatmap (R² Score)")
        plt.xlabel("Gamma")
        plt.ylabel("C")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "figure_10_svr_heatmap.png"))
        plt.close()
    except Exception as e:
        print(f"Could not generate SVR heatmap: {e}")


# -----------------
# MAIN SCRIPT
# -----------------

def main():
    # --- 1. Configuration & Setup ---
    TRAIN_CSV_PATH = "data/preprocessed_dataset.csv"
    TARGET_COL = "LN_IC50"  # Assumed to be log-transformed as in paper
    RANDOM_SEED = 42
    PLOTS_DIR = "plots"

    # Create directory for plots
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # --- 2. Load Data ---
    try:
        train_df = pd.read_csv(TRAIN_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: {TRAIN_CSV_PATH} not found.")
        print("Creating a dummy file: 'data/preprocessed_dataset.csv' to run this script.")
        # Create a dummy file for demonstration
        data = {'feature_1': np.random.rand(100),
                'feature_2_skewed': np.random.gamma(1, 2, 100) * 10,
                'feature_3': np.random.rand(100),
                'feature_4_skewed': np.random.gamma(2, 5, 100) * 5,
                'LN_IC50': np.random.rand(100) * 5}
        train_df = pd.DataFrame(data)
        train_df.to_csv(TRAIN_CSV_PATH, index=False)

    X, y = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]

    # --- 3. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # --- 4. Implement Paper's Pre-processing ---

    # Identify Highly Skewed Features (CS_F) from the training set
    # Using a skewness threshold > 2.0 as mentioned in the paper
    skew_vals = X_train.skew()
    cs_f_features = skew_vals[abs(skew_vals) > 2.0].index.tolist()
    print(f"Found {len(cs_f_features)} highly skewed features (CS_F): {cs_f_features}")

    # Create the 3 datasets as described in the paper

    # Dataset 1: Base (Just standard scaling)
    scaler_base = StandardScaler().fit(X_train)
    X_train_base = scaler_base.transform(X_train)
    X_test_base = scaler_base.transform(X_test)

    # [cite_start]Dataset 2: Enhanced (Exclude CS_F) [cite: 415-417, 875]
    X_train_enhanced = X_train.drop(columns=cs_f_features) # DataFrame
    X_test_enhanced = X_test.drop(columns=cs_f_features) # DataFrame
    scaler_enhanced = StandardScaler().fit(X_train_enhanced)
    X_train_enhanced_scaled = scaler_enhanced.transform(X_train_enhanced) # Numpy array
    X_test_enhanced_scaled = scaler_enhanced.transform(X_test_enhanced) # Numpy array

    # Dataset 3: Enhanced Log (Log-transform CS_F)
    X_train_enhanced_log = X_train.copy()
    X_test_enhanced_log = X_test.copy()

    # Apply np.log1p (handles 0 values) to the skewed features
    # This assumes original data is non-negative, like gene expression
    for col in cs_f_features:
        X_train_enhanced_log[col] = np.log1p(X_train_enhanced_log[col])
        X_test_enhanced_log[col] = np.log1p(X_test_enhanced_log[col])

    scaler_enhanced_log = StandardScaler().fit(X_train_enhanced_log)
    X_train_enhanced_log_scaled = scaler_enhanced_log.transform(X_train_enhanced_log)
    X_test_enhanced_log_scaled = scaler_enhanced_log.transform(X_test_enhanced_log)

    # Store datasets in a dictionary for iteration
    # This uses the bug fix:
    # 1. Passes the scaled arrays (e.g., X_train_enhanced_scaled) for training.
    # 2. Passes the column lists from the pre-scaled DataFrames (e.g., X_train_enhanced.columns).
    datasets = {
        "Base": (X_train_base, y_train, X_test_base, y_test, X_train.columns),
        "Enhanced": (X_train_enhanced_scaled, y_train, X_test_enhanced_scaled, y_test, X_train_enhanced.columns),
        "Enhanced Log": (X_train_enhanced_log_scaled, y_train, X_test_enhanced_log_scaled, y_test, X_train_enhanced_log.columns)
    }

    # --- 5. Define Models (including paper's PLS) ---
    # Determine n_components for PLS (must be <= n_features)
    # Use the smallest feature set (Enhanced) as the limiting factor
    max_components = X_train_enhanced.shape[1]
    pls_components = min(10, max_components) # Use 10 components or max available

    models_to_train = {
        "Linear Regression": LinearRegression(),
        "PLS Regression": PLSRegression(n_components=pls_components),
        "SVR": SVR(), # Will be grid-searched
        "Random Forest": RandomForestRegressor(random_state=RANDOM_SEED),
        "XGBoost": xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_SEED
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(100, 50), # Simplified for speed
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=RANDOM_SEED,
            early_stopping=True,
            n_iter_no_change=10
        )
    }

    # SVR parameters for GridSearchCV
    # Reduced grid for speed
    svr_param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01],
        'epsilon': [0.1, 0.2],
        'kernel': ['rbf']
    }

    # --- 6. Train Models & Evaluate ---
    all_scores = []
    trained_base_models = {}
    saved_svr_grid_search = None # To save the SVR grid search object for plotting

    for d_name, (X_tr, y_tr, X_te, y_te, features) in datasets.items():
        print(f"\n--- Training on '{d_name}' Dataset ---")

        # Adjust PLS components if necessary for this specific dataset
        if "PLS Regression" in models_to_train:
            current_max_components = X_tr.shape[1]
            models_to_train["PLS Regression"].n_components = min(pls_components, current_max_components)

        for m_name, model in models_to_train.items():
            print(f"Training {m_name}...")

            # Special handling for SVR GridSearch
            if m_name == "SVR":
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=svr_param_grid,
                    scoring='r2',
                    cv=3, # Reduced for speed
                    n_jobs=-1
                )
                grid_search.fit(X_tr, y_tr)
                trained_model = grid_search.best_estimator_
                print(f"Best SVR Params: {grid_search.best_params_}")

                # Save the grid search object from the 'Base' dataset
                if d_name == "Base":
                    saved_svr_grid_search = grid_search

            # Special handling for XGBoost to get eval_results
            elif m_name == "XGBoost":
                eval_set = [(X_tr, y_tr), (X_te, y_te)]
                model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
                trained_model = model

            # Standard training for other models
            else:
                trained_model = model.fit(X_tr, y_tr)

            # Evaluate on test set
            y_pred = trained_model.predict(X_te)

            # Calculate metrics
            r2 = r2_score(y_te, y_pred)
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            mae = mean_absolute_error(y_te, y_pred)

            print(f"Test Scores for {m_name} on {d_name}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

            # Store scores
            all_scores.append({"Dataset": d_name, "Model": m_name, "Metric": "R2", "Score": r2})
            all_scores.append({"Dataset": d_name, "Model": m_name, "Metric": "RMSE", "Score": rmse})
            all_scores.append({"Dataset": d_name, "Model": m_name, "Metric": "MAE", "Score": mae})

            # Save the models trained on the "Base" dataset
            if d_name == "Base":
                trained_base_models[m_name] = trained_model

    scores_df = pd.DataFrame(all_scores)

    # --- 7. Save Models ---
    with open("data/models.pkl", "wb") as f:
        pickle.dump(trained_base_models, f)
    print("\nSuccessfully trained and saved base models to 'data/models.pkl'")

    # --- 8. Generate Visualizations ---
    print("Generating visualizations...")

    # Fig 2
    plot_feature_distribution(X_train, cs_f_features, X_train_enhanced_log, PLOTS_DIR)

    # Fig 3
    plot_model_r2_comparison(scores_df, PLOTS_DIR)

    # Fig 4 (New Grid)
    plot_prediction_vs_actual_grid(
        trained_base_models,
        datasets["Base"][2], # X_test_base
        datasets["Base"][3], # y_test
        PLOTS_DIR
    )

    # Fig 5
    plot_random_forest_importance(
        trained_base_models['Random Forest'],
        datasets["Base"][4], # Base features
        PLOTS_DIR
    )

    # Fig 6
    plot_xgboost_convergence(trained_base_models['XGBoost'], PLOTS_DIR)

    # Fig 7
    plot_mlp_loss(trained_base_models['MLP'], PLOTS_DIR)

    # Fig 8
    plot_all_metric_comparison(scores_df, PLOTS_DIR)

    # Fig 9 (New)
    plot_residuals_grid(
        trained_base_models,
        datasets["Base"][2], # X_test_base
        datasets["Base"][3], # y_test
        PLOTS_DIR
    )

    # Fig 10 (New)
    if saved_svr_grid_search:
        plot_svr_heatmap(saved_svr_grid_search, PLOTS_DIR)
    else:
        print("Skipping SVR heatmap: Grid search object not saved.")

    print(f"All visualizations saved to '{PLOTS_DIR}/' directory.")


if __name__ == "__main__":
    main()