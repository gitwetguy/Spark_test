from operator import add
from pyspark import SparkContext , SparkConf
from pyspark . streaming import StreamingContext


conf = SparkConf ()
conf . setAppName ( 'TestDStream' )
conf . setMaster ( 'spark://master:7077' )
conf.set('spark.executor.memory', '2g')
conf.set('spark.executor.cores', '1')
#conf . setAppName ( 'TestDStream2' )
sc = SparkContext ( conf = conf )
ssc = StreamingContext ( sc , 5)
lines = ssc . textFileStream ( 'file:///home/master/spark_test/Spark_test/log' )
words = lines . flatMap ( lambda line : line . split ( '\n' ))
wordCounts = words . map ( lambda x : ( x , 1 )). reduceByKey ( add )
wordCounts . pprint ()
ssc . start ()
ssc . awaitTermination ()
