# mahout

mahout is a framework which should help to execute scalable performant machine learning applications.
It can use Hadoop, Spark or Flink. You can find an example how
mahout can be used on top of hadoop in 'association rules learning'

you can install mahout on mac by

    brew install mahout

Unluckily some algorthims are missing in the newer versions because of lacking maintainers.
You can download older versions, unpack them and start them locally.
I use mahout 0.8 in the example (http://archive.apache.org/dist/mahout/0.8/)
In order to use mahout parts in your java application, add this dependencies:

    compile 'org.apache.mahout:mahout-core:0.7'
    compile 'org.apache.mahout:mahout-math:0.7'


Mahout is a tool which is often used. A good example can be found under https://chimpler.wordpress.com/2013/05/02/finding-association-rules-with-mahout-frequent-pattern-mining/

Under data is a copy of that example included.
MahoutAssociationLearningExample.csv contains different baskets, let's execute it.

Install software:

* install hadoop
* use mahout 0.8, because the fpg algorithm is removed in later versions of mahout

Prepare dataformat and load file into hadoop hdfs:

* execute MahoutDataPrepare (this will generate MahoutAssociationLearningExampleOutput.dat - containing the transaction data in a different format and MahoutAssociationLearningExampleMapping.csv - containing the mapping between the item name and the item id
* $ hadoop fs -put MahoutAssociationLearningExampleOutput.dat MahoutAssociationLearningExampleOutput.dat

(it can be that you have to use hdfs dfs instead of hadoop fs here)

execute the algorithm:

* $ mahout fpg -i MahoutAssociationLearningExampleOutput.dat -o patterns -k 10 -method mapreduce -s 2
* if needed adapt parameters -k 10 means that for each item i, the top 10 association rule, -s 2 means only consider the sets of items which appears in more than 2 transactions
* look at raw results: $ mahout seqdumper -i patterns/frequentpatterns/part-r-00000   ([141, 0],42) means that the pair item 141 and item 0 appears in 42 transactions and so on

Let's take a look int the results

* $ hadoop fs -getmerge patterns/frequentpatterns frequentpatterns.seq
* $ hadoop fs -get patterns/fList fList.seq
* execute MahoutResultEvaluation

You will find something like:

    [Sweet Relish] => Hot Dogs: supp=0.047, conf=0.508, lift=5.959, conviction=1.859
    [Eggs] => Hot Dogs: supp=0.043, conf=0.460, lift=3.751, conviction=1.626
    ...

which means people who bought Sweet Relish often bought Hot Dogs
