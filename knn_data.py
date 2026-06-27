"""
Jacob Maurer
6/15/2026
Purpose: See if any data has decent clusters. Could be an extra feature to help encoding, or to see if something
Like SMOTE would help generate synthetic samples
"""
from import_dataset import open_csv_dataset, open_hopv15_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

def test_kmeans_silhouette(input_data, range_num_clusters=3):
        scores = []
        for i in range(3, range_num_clusters+1):
            print(f"Num Clusters Testing: {i}")
            classification = KMeans(i, random_state=42)
            predictions = classification.fit_predict(input_data)
            score = silhouette_score(input_data, predictions)
            scores.append(score)
        return scores

if __name__ == "__main__":
    # data = open_csv_dataset("./combined_data.csv")
    np.random.seed(42)
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    experimental_data = []
    for molecule in data:
        if np.isnan(molecule.experimental_data.HOMO) or np.isnan(molecule.experimental_data.LUMO) or np.isnan(molecule.experimental_data.optical_gap) or np.isnan(molecule.experimental_data.Voc) or np.isnan(molecule.experimental_data.Jsc) or np.isnan(molecule.experimental_data.PCE):
            continue
        experimental_data.append([molecule.experimental_data.HOMO, molecule.experimental_data.LUMO, molecule.experimental_data.optical_gap, molecule.experimental_data.Voc, molecule.experimental_data.Jsc, molecule.experimental_data.PCE])
    np.random.shuffle(experimental_data)
    experimental_data = np.asarray(experimental_data)
    silhouette_results = test_kmeans_silhouette(experimental_data, 30)
    for i, score in enumerate(silhouette_results):
        print(f"For {i+3} clusters, the silhouette score is {score:.4f}")
    