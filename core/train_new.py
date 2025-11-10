def main():
    # --- 1. Configuration & Setup ---
    TRAIN_CSV_PATH = "data/preprocessed_dataset.csv"
    TARGET_COL = "LN_IC50"
    RANDOM_SEED = 42
    PLOTS_DIR = "plots"

    # --- NEW: Skewness threshold from paper (lowered) ---
    # We'll use 1.5 to be more likely to catch features.
    SKEW_THRESHOLD = 1.5

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
                'feature_2_skewed': np.random.gamma(1, 2, 100) * 10, # Skew ~2.0
                'feature_3': np.random.rand(100),
                'feature_4_skewed': np.random.gamma(2, 5, 100) * 5,  # Skew ~1.4
                'LN_IC50': np.random.rand(100) * 5}
        train_df = pd.DataFrame(data)
        train_df.to_csv(TRAIN_CSV_PATH, index=False)

    X, y = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]

    # --- 3. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # --- 4. Implement Paper's Pre-processing ---

    # Identify Highly Skewed Features (CS_F) from the training set
    skew_vals = X_train.skew()

    # --- MODIFIED: Use the new threshold variable ---
    cs_f_features = skew_vals[abs(skew_vals) > SKEW_THRESHOLD].index.tolist()

    print(f"Found {len(cs_f_features)} features with skew > {SKEW_THRESHOLD}: {cs_f_features}")

    # (The rest of the pre-processing logic is the same)

    # Dataset 1: Base (Just standard scaling)
    scaler_base = StandardScaler().fit(X_train)
    X_train_base = scaler_base.transform(X_train)
    X_test_base = scaler_base.transform(X_test)

    # Dataset 2: Enhanced (Exclude CS_F)
    X_train_enhanced = X_train.drop(columns=cs_f_features) # DataFrame
    X_test_enhanced = X_test.drop(columns=cs_f_features) # DataFrame
    scaler_enhanced = StandardScaler().fit(X_train_enhanced)
    X_train_enhanced_scaled = scaler_enhanced.transform(X_train_enhanced) # Numpy array
    X_test_enhanced_scaled = scaler_enhanced.transform(X_test_enhanced) # Numpy array

    # Dataset 3: Enhanced Log (Log-transform CS_F)
    X_train_enhanced_log = X_train.copy()
    X_test_enhanced_log = X_test.copy()

    for col in cs_f_features:
        X_train_enhanced_log[col] = np.log1p(X_train_enhanced_log[col])
        X_test_enhanced_log[col] = np.log1p(X_test_enhanced_log[col])

    scaler_enhanced_log = StandardScaler().fit(X_train_enhanced_log)
    X_train_enhanced_log_scaled = scaler_enhanced_log.transform(X_train_enhanced_log)
    X_test_enhanced_log_scaled = scaler_enhanced_log.transform(X_test_enhanced_log)

    datasets = {
        "Base": (X_train_base, y_train, X_test_base, y_test, X_train.columns),
        "Enhanced": (X_train_enhanced_scaled, y_train, X_test_enhanced_scaled, y_test, X_train_enhanced.columns),
        "Enhanced Log": (X_train_enhanced_log_scaled, y_train, X_test_enhanced_log_scaled, y_test, X_train_enhanced_log.columns)
    }

    # --- 5. Define Models ---

    # Determine n_components for PLS (must be <= n_features)
    # Use the smallest feature set (Enhanced) as the limiting factor
    max_components = X_train_enhanced.shape[1]

    # --- MODIFIED: Set a safer default for pls_components ---
    # Set to min(10, num_features - 1) and ensure it's at least 1
    pls_components = min(10, max_components - 1)
    pls_components = max(1, pls_components) # Ensure n_components is at least 1

    models_to_train = {
        "Linear Regression": LinearRegression(),
        "PLS Regression": PLSRegression(n_components=pls_components), # Use safer default
        "SVR": SVR(),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_SEED),
        "XGBoost": xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_SEED
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=RANDOM_SEED,
            early_stopping=True,
            n_iter_no_change=10
        )
    }

    # SVR parameters for GridSearchCV
    svr_param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01],
        'epsilon': [0.1, 0.2],
        'kernel': ['rbf']
    }

    # --- 6. Train Models & Evaluate ---
    all_scores = []
    trained_base_models = {}
    saved_svr_grid_search = None

    for d_name, (X_tr, y_tr, X_te, y_te, features) in datasets.items():
        print(f"\n--- Training on '{d_name}' Dataset ---")

        for m_name, model in models_to_train.items():

            # --- MODIFIED: Add robust check for PLS n_components ---
            if m_name == "PLS Regression":
                # Ensure n_components is ALWAYS < n_features for the CURRENT dataset
                current_max_components = X_tr.shape[1]
                n_comps = min(pls_components, current_max_components - 1)
                n_comps = max(1, n_comps) # Ensure at least 1

                # Update the model's n_components *before* training
                model.n_components = n_comps

            print(f"Training {m_name}...")

            # Special handling for SVR GridSearch
            if m_name == "SVR":
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=svr_param_grid,
                    scoring='r2',
                    cv=3,
                    n_jobs=-1
                )
                grid_search.fit(X_tr, y_tr)
                trained_model = grid_search.best_estimator_
                print(f"Best SVR Params: {grid_search.best_params_}")

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

            all_scores.append({"Dataset": d_name, "Model": m_name, "Metric": "R2", "Score": r2})
            all_scores.append({"Dataset": d_name, "Model": m_name, "Metric": "RMSE", "Score": rmse})
            all_scores.append({"Dataset": d_name, "Model": m_name, "Metric": "MAE", "Score": mae})

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