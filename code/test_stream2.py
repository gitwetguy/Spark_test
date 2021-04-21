from operator import add
from pyspark import SparkContext , SparkConf
from pyspark . streaming import StreamingContext

def sliding(rdd, n):
    assert n > 0
    def gen_window(xi, n):
        x, i = xi
        return [(i - offset, (i, x)) for offset in xrange(n)]

    return (
        rdd.
        zipWithIndex(). # Add index
        flatMap(lambda xi: gen_window(xi, n)). # Generate pairs with offset
        groupByKey(). # Group to create windows
        # Sort values to ensure order inside window and drop indices
        mapValues(lambda vals: [x for (i, x) in sorted(vals)]).
        sortByKey(). # Sort to makes sure we keep original order
        values(). # Get values
        filter(lambda x: len(x) == n)) # Drop beginning and end

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
print(type(words))
wordCounts = words . map ( lambda x : ( x , 1 )). reduceByKey ( add )
words . pprint ()
ssc . start ()
ssc . awaitTermination ()
