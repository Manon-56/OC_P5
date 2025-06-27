#Script dans lequel j'ai défini différentes méthodes qui seront importées dans les notebooks
from sqlite3 import connect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
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

	SELECT c.customer_id,
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
	data = pd.read_sql(sql_query, conn)
	print(len(data))
	return data



def prepare_compute_evaluate_kmeans(df: pd.DataFrame, data_description: str, max_num_clusters: int, random_state: int):

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