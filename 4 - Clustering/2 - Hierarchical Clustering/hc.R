# Hierarchial Clustering

# Importing the Mall dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

# Using dendrogram to find optimal number of clusters
dendrogram = hclust(dist(X, method = 'euclidean'),
                    method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean Distance')

# Fitting the hiearchial clustering to the dataset
hc = hclust(dist(X, method = 'euclidean'),
                    method = 'ward.D')
y_hc = cutree(hc, 5)

# Visualising the clusters
# install.packages('cluster')
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Cluster of Clients'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')