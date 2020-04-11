# Natural Language Processing

# Importing the dataset
# stringsAsFactors is passed FALSE so that each string is not treated as a factor in R
dataset_original = read.delim('Restaurant_Reviews.tsv', 
                     quote = '', 
                     stringsAsFactors = FALSE)

# Cleaning the texts
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Create the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classifier to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)

true_positives = cm[1, 1]
false_positives = cm[1 , 2]
false_negatives = cm[2, 1]
true_negatives = cm[2, 2]

# Performance Matrices for the classification models
model_accuracy = ( true_positives + true_negatives ) / ( true_positives + true_negatives + false_positives + false_negatives )
model_precision = true_positives / ( true_positives + false_positives )
model_recall = true_positives / ( true_positives + false_negatives )
model_f1_score = 2 * model_precision * model_recall / ( model_precision + model_recall )
