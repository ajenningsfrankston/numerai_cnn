
#
# re-order the features so that correlated features are grouped together
#

from sklearn.cluster import KMeans

def reorder_features(features):
    kmeans = KMeans(n_clusters=5, random_state=0).fit(features)
    clusters = kmeans.predict(features)