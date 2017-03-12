#Import the necessary modules to run pyspark, and the libraries used
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
import numpy as np

#Enter configuration
conf = SparkConf().setMaster("local").setAppName("LRG")

#Create spark context
sc = SparkContext(conf = conf)

#Load and examine the data. I remove any NA values for horsepower.
path = "auto_mpg_original.csv"
raw_data = sc.textFile(path)
num_data = raw_data.count()
records = raw_data.map(lambda x: x.split(",")).filter(lambda r: r[3] != 'NA')
first = records.first()
print first
print num_data

#Function to extract feature (displacement)
def extract_features(record):
    return [float(record[2])]

#Function to extract target variable (horsepower)
def extract_label(record):
    return float(record[3])

#Create LabeledPoint with displacement and horsepower
data = records.map(lambda r: LabeledPoint(extract_label(r),extract_features(r)))

#Split the dataset into training (80%) and testing (20%) for LinearRegression
data_with_idx = data.zipWithIndex().map(lambda (k, v): (v, k))
test = data_with_idx.sample(False, 0.2, 42)
train = data_with_idx.subtractByKey(test)

train_data = train.map(lambda (idx, p): p)
test_data = test.map(lambda (idx, p) : p)
train_size = train_data.count()
test_size = test_data.count()
print "Training data size: %d" % train_size
print "Test data size: %d" % test_size
print "Total data size: %d " % num_data
print "Train + Test size : %d" % (train_size + test_size)


print "First 5 train data records: " + str(train_data.take(5))
print "First 5 test data records: " + str(test_data.take(5))

#Train the model from the train data
linear_model = LinearRegressionWithSGD.train(train_data,iterations = 500,step = 0.000001,regParam = 0.0, regType= 'l2', intercept = False)
print linear_model

#Show a few predictions
true_vs_predicted = test_data.map(lambda p: (p.label, linear_model.predict(p.features)))
print "First 5 Linear Model predictions: " + str(true_vs_predicted.take(5))

#Function to calculate absolute error
def abs_error(actual, pred):
    return np.abs(pred - actual)

#Function to calculate squared log error
def squared_log_error(pred, actual):
    return (np.log(pred + 1) - np.log(actual + 1))**2

#Calculate RMSLE and MAE and print the relevant values
rmsle = np.sqrt(true_vs_predicted.map(lambda (t, p): squared_log_error(t,p)).mean())
mae = true_vs_predicted.map(lambda (t, p): abs_error(t, p)).mean()

print "Linear Model - Mean Absolute Error: %2.4f" % mae
print "Linear Model - Root Mean Squared Log Error: %2.4f" % rmsle
