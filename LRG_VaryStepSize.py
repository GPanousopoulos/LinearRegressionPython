#Import the necessary modules to run pyspark, and the libraries used
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
import numpy as np

#Enter configuration
conf = SparkConf().setMaster("local").setAppName("LRG")

#Create spark context
sc = SparkContext(conf = conf)

#Load and examine the data. I remove any NA values.
path = "auto_mpg_original.csv"
raw_data = sc.textFile(path)
num_data = raw_data.count()
records = raw_data.map(lambda x: x.split(",")).filter(lambda r: 'NA' not in r)
first = records.first()
print first
print num_data

#Cache the RDD
records.cache()

#Define the mappings function
def get_mapping(rdd, idx):
    return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()

#Define categorical and numerical features
mappings = [get_mapping(records, i) for i in [1,6,7]]
cat_len = sum(map(len, mappings))
num_len = len(records.first()[2:5])
total_len = num_len + cat_len

print "Feature vector length for categorical features: %d" % cat_len
print "Feature vector length for numerical features: %d" % num_len
print "Total feature vector length: %d" % total_len


#Function to extract features (all except car name and mpg) and apply 1-of-k to the categorical ones
def extract_features(record):
    cat_vec = np.zeros(cat_len)
    i = 0
    step = 0
    k = [1,6,7]
    for field in [record[x] for x in k]:
        m = mappings[i]
        idx = m[field]
        cat_vec[idx + step] = 1
        i = i + 1
        step = step + len(m)
    num_vec = np.array([float(field) for field in record[2:5]])
    return np.concatenate((cat_vec, num_vec))

#Function to extract target variable (mpg)
def extract_label(record):
    return float(record[0])

#Create LabeledPoint with mpg as the label
data = records.map(lambda r: LabeledPoint(extract_label(r),extract_features(r)))

#Split the dataset into training and testing for LinearRegression
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

#Function to calculate squared log error
def squared_log_error(pred, actual):
    return (np.log(pred + 1) - np.log(actual + 1))**2


def evaluate(train, test, iterations, step, regParam,regType,intercept):
    model = LinearRegressionWithSGD.train(train,iterations,step,regParam=regParam, regType=regType, intercept=intercept)
    tp = test.map(lambda p: (p.label, model.predict(p.features)))
    rmsle = np.sqrt(tp.map(lambda (t, p): squared_log_error(t,p)).mean())
    return rmsle

#Test different values for step size and see how performance improves
params = [0.000001, 0.005, 0.01, 0.05, 0.1, 0.5]
metrics = [evaluate(train_data, test_data, 100, param, 0.0, 'l2', False) for param in params]
print params
print metrics