import sys

from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext

if __name__ == "__main__":
      conf = SparkConf ()
      conf . setAppName ( 'TestDsocketStream' )
      conf . setMaster ( 'spark://master:7077' )
      conf.set('spark.executor.memory', '2g')
      conf.set('spark.executor.cores', '1')
      if len(sys.argv) != 3:
            print("Usage: network_wordcount.py <hostname> <port>", file=sys.stderr)
            sys.exit(-1)
      sc = SparkContext(conf = conf)
      ssc = StreamingContext(sc, 5)

      lines = ssc.socketTextStream(sys.argv[1], int(sys.argv[2]))
      counts = lines.flatMap(lambda line: line.split(" "))\
                              .map(lambda word: (word, 1))\
                              .reduceByKey(lambda a, b: a+b)
      #print(lines)
      data = lines.count()
      counts.pprint()
      ssc.start()
      ssc.awaitTermination()