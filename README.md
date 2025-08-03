# 🧠 Image Clustering with K-Means and DBSCAN (NumPy + Matplotlib)

This project demonstrates unsupervised image clustering using custom implementations of **K-Means** and **DBSCAN** algorithms from scratch with **NumPy**. The project avoids external ML libraries (like scikit-learn) except for input validation and focuses on understanding the inner workings of clustering techniques.

## 📁 Project Structure

- `clustering_algorithms.py` — Custom `CustomKMeans` and `CustomDBSCAN` classes with `fit` and `fit_predict` methods
- `clustering_task.ipynb` — Notebook for applying the clustering algorithms on sample image data and visualizing the results using matplotlib

## 🚀 Features

- From-scratch implementation of:
  - K-Means clustering
  - DBSCAN clustering
- Compatible with basic scikit-learn interface (`fit`, `fit_predict`)
- Image or feature data clustering
- Visualizations using matplotlib

## 🔧 Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Jupyter Notebook (optional but recommended)

Install dependencies:
```bash
pip install numpy matplotlib notebook
