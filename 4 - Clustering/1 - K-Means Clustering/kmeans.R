# K-Means Clustering

# Importing the Mall dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

# Using Elbow Method to find optimal number of clusters
set.seed(6)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(X, i)$withins)
plot(1:10, 
  wcss, 
  type = 'b', 
  main = paste('Clustes of Clients'), 
  xlab = 'Number of Clusters',
  ylab = 'WCSS')

# Applying K-Means to Mall Dataset
set.seed(29)
kmeans = kmeans(X, 5, iter.max = 300, nstart = 10)

# Visualising the Clusters
# install.packages('cluster')
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Cluster of Clients'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')