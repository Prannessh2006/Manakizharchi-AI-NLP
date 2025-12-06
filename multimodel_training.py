X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train[:, :1], y_train_reg)  # simple linear on PC1
pred_lr = lr.predict(X_test[:, :1])
print("Simple Linear Regression metrics:")
print("MAE", mean_absolute_error(y_test_reg, pred_lr))
print("MSE", mean_squared_error(y_test_reg, pred_lr))
print("RMSE", np.sqrt(mean_squared_error(y_test_reg, pred_lr)))
print("R2", r2_score(y_test_reg, pred_lr))

mlr = LinearRegression()
mlr.fit(X_train, y_train_reg)
pred_mlr = mlr.predict(X_test)
print("Multiple Linear Regression R2:", r2_score(y_test_reg, pred_mlr))

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train[:, :5])
X_test_poly = poly.transform(X_test[:, :5])
poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train_reg)
pred_poly = poly_lr.predict(X_test_poly)
print("Polynomial Regression R2:", r2_score(y_test_reg, pred_poly))

Xc_train, Xc_test, y_c_train, y_c_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
logreg = LogisticRegression(max_iter=300)
logreg.fit(Xc_train, y_c_train)
pred_logreg = logreg.predict(Xc_test)
print("Logistic Regression classification report:")
print(classification_report(y_c_test, pred_logreg))

models = {
    "NaiveBayes": MultinomialNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3), # Changed n_neighbors to 3
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Note: MultinomialNB expects count-like features (we have TF-IDF). We'll create a CountVectorizer version.
count_vec = CountVectorizer(max_features=5000, ngram_range=(1,2))
X_count = count_vec.fit_transform(df['clean_joined'])
Xc_train_cv, Xc_test_cv, y_c_train_cv, y_c_test_cv = train_test_split(X_count, y_clf, test_size=0.2, random_state=42)

results = {}
for name, model in models.items():
    if name == "NaiveBayes":
        m = model
        m.fit(Xc_train_cv, y_c_train_cv)
        preds = m.predict(Xc_test_cv)
        probs = m.predict_proba(Xc_test_cv)[:,1]
    else:
        # for other models use PCA features (scaled)
        m = model
        m.fit(Xc_train, y_c_train)  # using PCA features
        preds = m.predict(Xc_test)
        if hasattr(m, "predict_proba"):
            probs = m.predict_proba(Xc_test)[:,1]
        else:
            probs = None
    results[name] = {
        "accuracy": accuracy_score(y_c_test if name!="NaiveBayes" else y_c_test_cv, preds),
        "precision": precision_score(y_c_test if name!="NaiveBayes" else y_c_test_cv, preds),
        "recall": recall_score(y_c_test if name!="NaiveBayes" else y_c_test_cv, preds),
        "f1": f1_score(y_c_test if name!="NaiveBayes" else y_c_test_cv, preds)
    }
    print(f"{name} metrics:", results[name])

bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=42)
bag.fit(Xc_train, y_c_train)
pred_bag = bag.predict(Xc_test)
print("Bagging accuracy:", accuracy_score(y_c_test, pred_bag))

ada = AdaBoostClassifier(n_estimators=50, random_state=42)
ada.fit(Xc_train, y_c_train)
print("AdaBoost accuracy:", accuracy_score(y_c_test, ada.predict(Xc_test)))

gboost = GradientBoostingClassifier(n_estimators=100, random_state=42)
gboost.fit(Xc_train, y_c_train)
print("GradientBoosting accuracy:", accuracy_score(y_c_test, gboost.predict(Xc_test)))

mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42)
mlp.fit(Xc_train, y_c_train)
pred_mlp = mlp.predict(Xc_test)
print("MLP accuracy:", accuracy_score(y_c_test, pred_mlp))
print(classification_report(y_c_test, pred_mlp))

kf = KFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42),
                            Xc_train, y_c_train, cv=kf, scoring='f1')
print("RandomForest CV F1 scores:", cv_scores, "mean:", cv_scores.mean())

k = 4 
km = KMeans(n_clusters=k, random_state=42)
clusters = km.fit_predict(X_pca)
df['cluster_kmeans'] = clusters

print(df['cluster_kmeans'].value_counts())

# Hierarchical clustering (agglomerative)
agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
agg_labels = agg.fit_predict(X_pca)
df['cluster_agg'] = agg_labels
print(df['cluster_agg'].value_counts())

tfidf_feat = tfidf.get_feature_names_out()
centroids = km.cluster_centers_

X_tfidf_arr = X_tfidf.toarray()
for i in range(k):
    idxs = np.where(clusters == i)[0]
    if len(idxs) == 0: continue
    mean_tfidf = X_tfidf_arr[idxs].mean(axis=0)
    top_idx = mean_tfidf.argsort()[-10:][::-1]
    top_terms = [tfidf_feat[j] for j in top_idx]
    print(f"Cluster {i} top terms:", top_terms)


transactions = df['tokens'].apply(lambda toks: list(set([t for t in toks if len(t) > 1]))).tolist()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
trans_df = pd.DataFrame(te_ary, columns=te.columns_)


freq_itemsets = apriori(trans_df, min_support=0.01, use_colnames=True)
rules = association_rules(freq_itemsets, metric="lift", min_threshold=1.2)
print("Top association rules (sorted by lift):")
print(rules.sort_values('lift', ascending=False).head(10))

import joblib
os.makedirs('models', exist_ok=True)
joblib.dump(tfidf, 'models/tfidf.pkl')
joblib.dump(count_vec, 'models/count_vec.pkl')
joblib.dump(pca, 'models/pca.pkl')
joblib.dump(mlr, 'models/multiple_linear_regression.pkl')
joblib.dump(logreg, 'models/logistic_regression.pkl')
joblib.dump(mlp, 'models/mlp_classifier.pkl')
joblib.dump(km, 'models/kmeans.pkl')
joblib.dump(bag, 'models/bagging.pkl')
print("Models saved to /models")

df.to_csv('data/processed_reviews.csv', index=False)
print("Processed CSV saved to data/processed_reviews.csv")
