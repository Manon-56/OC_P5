#Script dans lequel j'ai défini différentes méthodes qui seront importées dans les notebooks
from sqlite3 import connect
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from hdbscan.validity import validity_index
from tqdm import tqdm


def import_data(date_fin: str):
	"""
	Cette méthode permet d'importer les données de chaque client à partir de la base de données et de les stocker dans un DataFrame. 
	Input :
		date_fin = date au format '%Y-%m-%d' considérée comme le présent i.e. date après laquelle les données ne sont plus importées
	Output :
		data : DataFrame contenant la liste des clients et les variables associées
	"""
	sql_query = f"""
	WITH favourite_product AS (
		SELECT customer_id,
			   product_category_name_english AS type_produit_prefere
		FROM (
			SELECT o.customer_id,
				   p.product_category_name,
				   t.product_category_name_english,
				   count(p.product_id) AS nb_products_by_category,
				   ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY COUNT(p.product_id) DESC) AS rang
			FROM orders o
			JOIN order_items i ON i.order_id = o.order_id
			JOIN products p ON p.product_id = i.product_id
			JOIN translation t ON p.product_category_name = t.product_category_name
			GROUP BY o.customer_id,
					 p.product_category_name,
					 t.product_category_name_english
		)
		WHERE rang = 1
	),

	paiement_par_commande AS (
		SELECT order_id,
			   SUM(payment_value) AS depense_commande
		FROM order_pymts
		GROUP BY order_id
	),

	items_par_commande AS (
		SELECT order_id,
			   COUNT(order_item_id) AS nb_produits_commande,
			   SUM(price) AS prix_produit_commande,
			   SUM(freight_value) AS prix_transport_commande
		FROM order_items
		GROUP BY order_id
	),

	reviews_par_commande AS (
		SELECT order_id,
			   AVG(review_score) AS score_moyen_commande
		FROM order_reviews
		GROUP BY order_id
	)

	SELECT c.customer_unique_id,
		   c.customer_state,
		   COUNT(DISTINCT o.order_id) AS nb_commandes,
		   SUM(i.nb_produits_commande) AS nb_produits,
		   SUM(i.prix_produit_commande) AS prix_produit_total,
		   SUM(i.prix_transport_commande) AS prix_transport_total,
		   SUM(p.depense_commande) AS depense_totale,
		   (julianday('{date_fin}') - julianday(MAX(o.order_purchase_timestamp))) AS recence,
		   AVG(r.score_moyen_commande) AS score_moyen,
		   f.type_produit_prefere
	FROM customers c
	JOIN orders o ON c.customer_id = o.customer_id
	LEFT JOIN items_par_commande i ON i.order_id = o.order_id
	LEFT JOIN reviews_par_commande r ON o.order_id = r.order_id
	LEFT JOIN paiement_par_commande p ON o.order_id = p.order_id
	LEFT JOIN favourite_product f ON f.customer_id = c.customer_id
	GROUP BY c.customer_unique_id
	HAVING recence>=0;
	"""
	conn = connect(database='olist.db')
	# data = pd.read_sql(sql_query, conn)
	data = pl.read_database(sql_query, conn)
	print(len(data))
	return data



def prepare_compute_evaluate_kmeans(df: pd.DataFrame, data_description: str, max_num_clusters: int, random_state: int = 42):

	#Transform dataframe into polars for memory usage
	df = pl.from_pandas(df)
	# Préparation des données
	pt = PowerTransformer(method="yeo-johnson")
	X_transformed = pt.fit_transform(df)

	# Calcul des métriques pour différents nombres de clusters
	range_clusters = range(2, max_num_clusters+1)
	silhouette_scores = []
	db_scores = []
	inertia_values = []

	for n_clusters in tqdm(range_clusters):
		kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
		cluster_labels = kmeans.fit_predict(X_transformed)
		
		print("Calcul du score de silhouette ...")
		silhouette_avg = silhouette_score(X_transformed, cluster_labels)
		print("Calcul TERMINE")
		
		print("Calcul du score de Davies-Bouldin ...")
		db_avg = davies_bouldin_score(X_transformed, cluster_labels)
		print("Calcul TERMINE")
		
		inertia = kmeans.inertia_
		silhouette_scores.append(silhouette_avg)
		db_scores.append(db_avg)
		inertia_values.append(inertia)

	# Visualisation des résultats
	plt.figure(figsize=(15, 5))

	# Graphique inertie
	plt.subplot(131)
	plt.plot(range_clusters, inertia_values, "b-", marker="o")
	plt.title("Méthode du coude")
	plt.xlabel("Nombre de clusters")
	plt.ylabel("Inertie")

	# Graphique Davies-Bouldin
	plt.subplot(132)
	plt.plot(range_clusters, db_scores, "r-", marker="o")
	plt.title("Score Davies-Bouldin")
	plt.xlabel("Nombre de clusters")
	plt.ylabel("DB Index")

	# Graphique Silhouette
	plt.subplot(133)
	plt.plot(range_clusters, silhouette_scores, "g-", marker="o")
	plt.title("Score Silhouette")
	plt.xlabel("Nombre de clusters")
	plt.ylabel("Silhouette Score")

	plt.suptitle(data_description)
	plt.tight_layout()
	plt.show()

	# Sélection du meilleur nombre de clusters
	best_k_inertia = range_clusters[np.argmax(np.gradient(inertia_values))]
	best_k_db = range_clusters[np.argmin(db_scores)]
	best_k_silhouette = range_clusters[np.argmax(silhouette_scores)]

	print(f"{data_description}\nRecommandations pour K basées sur les différentes méthodes d'évaluation:")
	print(f"Méthode du coude: K = {best_k_inertia}")
	print(f"Davies-Bouldin: K = {best_k_db}")
	print(f"Silhouette: K = {best_k_silhouette}")


def prepare_compute_evaluate_dbscan(data: pd.DataFrame, data_description: str, eps: float, min_samples: int):
	"""
	Outputs : 
	- Validity index : This is a numeric value between -1 and 1, with higher values indicating a ‘better’ clustering.
	"""
	#mark beginning of process
	print(f"# {data_description}")

	#scale data
	pt = PowerTransformer(method="yeo-johnson", standardize = True)
	X_transformed = pt.fit_transform(data)

	#compute clustering
	clustering = DBSCAN(
		eps=eps, min_samples=min_samples, metric="euclidean", algorithm="ball_tree"
	).fit(X_transformed)
	clusters = clustering.labels_
	print(f"Avec les paramètres choisis, DBSCAN a défini {len(set(clusters))} clusters")

	#evaluate clustering through outliers rate
	mask = (clusters == -1)
	percent_outliers = np.round(len(X_transformed[mask]) / len(X_transformed) * 100, 4)
	print(f"Il y a {percent_outliers}% d'outliers")

	# Enlever les points marqués comme bruit (label = -1)
	mask = clusters != -1
	X_WO_outliers = X_transformed[mask]
	labels_WO_outliers = clusters[mask]

	# Score spécifique aux algorithmes de densité
	validity = validity_index(X_WO_outliers, labels_WO_outliers, metric="euclidean")
	print(f"Density-based cluster validity : {validity}")

	# Moyennes par cluster
	unique_labels = np.unique(labels_WO_outliers)
	cluster_means = []
	for label in unique_labels:
		cluster_points = X_WO_outliers[labels_WO_outliers == label]
		cluster_means.append(cluster_points.mean(axis=0))
	cluster_means = np.array(cluster_means)

	# Radar plot
	num_vars = X_WO_outliers.shape[1]
	angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
	angles += angles[:1]  # fermeture du polygone


	# Plot
	fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

	for i, row in enumerate(cluster_means):
		row_closed = np.concatenate([row, [row[0]]])
		ax.plot(angles, row_closed, label=f"Cluster {unique_labels[i]}")
		ax.fill(angles, row_closed, alpha=0.2)

	# Labels des axes
	feature_labels = [f"{data.columns[i]}" for i in range(num_vars)]
	angles_labels = angles[:-1]  # enlever l'angle du doublon
	ax.set_xticks(angles_labels)
	ax.set_xticklabels(feature_labels)

	ax.set_title("Profil moyen par cluster (DBSCAN)", y=1.08)
	ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
	plt.tight_layout()
	plt.show()
