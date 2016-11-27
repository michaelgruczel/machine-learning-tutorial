# machine-learning-tutorial

This is an incomplete list of machine learning algorithms.
For each algorithm I try to list:

* general idea of the algorithm
* manual calculated example and / or java example
* use cases

This is just my private learning list, don't expect correctness or completeness.

## code examples and toolings

Let's first speak about some tools/frameworks often used for machine learning

### weka

weka is an open source machine learning library which includes a UI as well. 
I use the library for some examples here
See http://www.cs.waikato.ac.nz/ml/weka/ for full details

The dependency is:

    compile 'nz.ac.waikato.cms.weka:weka-stable:3.6.6'

### hadoop

hadoop is a group of tools, the most known parts are the hadoop file system (HDFS) and the map reduce framework.
HDFS allows a smart distribution of of huge files to several servers (by splitting them) and map reduce allows the smart spitting of tasks to several host 
and the re-integration/merge of the distributed results.
Hadoop is designed for running on several hosts. There is one master node/name node and as much slaves as you want.
In order to work properly key based ssh must be configured.
For the test purposes of this tutorial a local installation is good enough. 

Install 

I will give an example for a local single node installation on mac (with homebrew):

    brew install hadoop

There are 2 ways to use hadoop locally:

* you can run a single node cluster locally
* you can just use the libs directly.

If you want to use the second option, then skip the next lines and continue at "simple usage".
The first option is far more complicated, but offers more support in terms of
using the api and so on. In case you want to do it the hard way, you have to do this:
    
ensure ssh login locally works:
    
    // enable ssh login to localhost
    ssh localhost
    // needed steps can be:
    // 1. Enable Remote Login in “System Preferences” -> “Sharing”. Check “Remote Login”
    // 2. if needed, means ~/.ssh/id_rsa.pub is empty
    ssh-keygen -t rsa
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
    
configure it:
    
    /usr/local/Cellar/hadoop/2.7.2/bin/hdfs namenode -format
    
/usr/local/Cellar/hadoop/2.7.2/libexec/etc/hadoop/core-site.xml:

    <configuration>  
      <property>
        <name>hadoop.tmp.dir</name>
        <value>/usr/local/Cellar/hadoop/hdfs/tmp</value>
        <description>A base for other temporary directories.</description>
      </property>
      <property>
        <name>fs.default.name</name>                                     
        <value>hdfs://localhost:9000</value>                             
      </property>
    </configuration>    

/usr/local/Cellar/hadoop/2.7.2/libexec/etc/hadoop/mapred-site.xml:

    <configuration>
      <property>
        <name>mapred.job.tracker</name>
        <value>localhost:9010</value>
      </property>
    </configuration>


/usr/local/Cellar/hadoop/2.7.2/libexec/etc/hadoop/hdfs-site.xml:

    <configuration>
      <property>
        <name>dfs.replication</name>
        <value>1</value>
      </property>
    </configuration>

folder rights to upload files into hdfs

    hdfs dfs -mkdir /user
    hdfs dfs -mkdir /user/<username>
  
start
  
    /usr/local/Cellar/hadoop/2.7.2/sbin/start-dfs.sh
    
check http://localhost:50070/ to see it running   

in order to stop it later use:

    /usr/local/Cellar/hadoop/2.7.2/sbin/stop-dfs.sh    

for a more advanced config see for example https://amodernstory.com/2014/09/23/installing-hadoop-on-mac-osx-yosemite/
or https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html

simple usage:

the HDFS is then an own filesystem on your local system. 
In order to see the files whoch are stored in the filesystem execute:

    hadoop fs -ls
    
Let's store a file into the HDFS

    // in case you run a single node cluster
    hdfs dfs -put data/HadoopLoremIpsumExample HadoopLoremIpsumExampleInHDFS
    
or    
    
    // if you only use the libs
    hadoop fs -put file ./data/HadoopLoremIpsumExample HadoopLoremIpsumExampleInHDFS   
    
The second important part of hadoop is map reduce. On base of this map reduce you can write java programs and embed them into hadoop.
Let's do this with the official example.
See HadoopExampleWordCount in java-examples to get an impression how map reduce works in java.
Let's now package it as a jar.

    cd java-examples
    ./gradlew build
    cp build/libs/java-examples.jar ./..
    cd ..
    mv java-examples.jar HadoopExampleWordCount.jar

Let's use it in hadoop:

    hadoop jar HadoopExampleWordCount.jar HadoopExampleWordCount HadoopLoremIpsumExampleInHDFS HadoopLoremIpsumExampleResultInHDFS

Let's move the result from the hdfs filesystem to the local disk (and merge the different reduce tasks to one file)

    hadoop fs -getmerge HadoopLoremIpsumExampleResultInHDFS ./data/HadoopLoremIpsumExampleResult

### mahout

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

### Yarn

Apache Hadoop NextGen MapReduce (YARN) / MapReduce 2.0 (MRv2)

The Hadoop system was based on the idea to
distribute the data with the assumptions that moving the algorithm is always cheaper.
Yarn is an improvement on top of hadoop MapReduce which adds a smart calculation (ResourceManager RM) 
of calculation resources in order to distribute the calculation in a smart way. That massively improves performance.

### Spring XD

Spring XD an extensible system for real time data ingestion and processing.
Spring XD application consists of inputs, processors and sinks which can be connected to streams.
There are defaults inputs, processors and sinks, but you can write your own ones.

Let's start it:

* Download https://repo.spring.io/libs-release/org/springframework/xd/spring-xd/1.3.1.RELEASE/spring-xd-1.3.1.RELEASE-dist.zip 
* Unzip the distribution
* cd spring-xd-1.3.1.RELEASE
* ./xd/bin/xd-singlenode
* check http://localhost:9393/admin-ui

Lets create an stream:

    ./shell/bin/xd-shell
    xd:>stream list
    xd:>stream create --name myStreamFromFile --definition "tail --name=/tmp/xdin | file"
    xd:>stream deploy --name myStreamFromFile
    echo "the good and the ugly" >> /tmp/xdin
    tail /tmp/xd/output/myStreamFromFile.out
    xd:>stream undeploy -name myStreamFromFile
    xd:>stream destroy -name myStreamFromFile

stop everything, lets now create a input which reads from a file, 
a custom processor which checks for the terms god and bad in the string in order to evaluate
whether the sentences are nice ones (I know this example is stupid, but it shows the core principle)

The example processor, you can find in the java-examples under SpringXDExampleProcessor,
let's bundle it into a jar

    cd java-examples
    ./gradlew build
    cp build/libs/java-examples.jar ./..
    cd ..
    mv java-examples.jar springxdexampleprocessor.jar

We have to make the definition of it public for spring xd

springxdexampleprocessor.xml:

    <?xml version="1.0" encoding="UTF-8"?>
    
    <beans:beans xmlns="http://www.springframework.org/schema/integration"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xmlns:beans="http://www.springframework.org/schema/beans"
      xsi:schemaLocation="http://www.springframework.org/schema/beans
        http://www.springframework.org/schema/beans/spring-beans.xsd
        http://www.springframework.org/schema/integration
        http://www.springframework.org/schema/integration/spring-integration.xsd">
      <channel id="input"/>
    
      <transformer input-channel="input" output-channel="output">
        <beans:bean class="SpringXDExampleProcessor" />
      </transformer>
    
      <channel id="output"/>
    </beans:beans>
    
deploy our custom processor and definition to xd installation:

    create folder ${xd.home}/modules/processor/springxdexampleprocessor
    create folder ${xd.home}/modules/processor/springxdexampleprocessor/config
    create folder ${xd.home}/modules/processor/springxdexampleprocessor/lib
    copy the springxdexampleprocessor.xml file to t${xd.home}/modules/processor/springxdexampleprocessor/config directory. 
    copy jar to ${xd.home}/modules/processor/springxdexampleprocessor/lib directory. 

    xd:>stream create --name myStreamFromFile2 --definition "tail --name=/tmp/xdin | springxdexampleprocessor | file"
    xd:>stream deploy --name myStreamFromFile2
    echo "the good and the ugly" >> /tmp/xdin
    tail /tmp/xd/output/myStreamFromFile2.out

You can image to chain several processors in this way to build a pipeline.

## Decision Tree Learning

The idea is to create a model that predicts a result based on several input variables.
Nodes correspond input variables and edges lead to possible results of each variable. 
A leaf represents a probability that all results of the path from root to the leave is true.
Decision trees are a type of supervised learning algorithms which can be used to predict outcomes.

One algorithm is C4.5. It selects as nodes the variables which most effectively split its set of samples into subsets.
This is done by using as splitting criterion the normalized information gain (which is difference in entropy).
The expected information gain is the change in information entropy H from a prior state to a state that takes some information as given.
It is used recursively from the root to the leafs.

![example](https://upload.wikimedia.org/wikipedia/commons/f/f3/CART_tree_titanic_survivors.png "example")

    Entropy = - p(a)*log(p(a)) - p(b)*log(p(b))
    Information gain =  IG(T,a) = H(T) - H(T|a) 

So in general:

* calculate all entropies for all variables and possible results.
* Now calculate all entropies for all variables under the condition that a result is given.
* Now you have the informations gains and you can select the highest one
* do this recursively until you reach the leaves 
 
There are  other algorithm and other metrics, check https://en.wikipedia.org/wiki/Decision_tree_learning for more information for more details

One example is explained here http://technobium.com/decision-trees-explained-using-weka/
You can find the code for it in this repository as well (WekaDecisionTreeApplication).
It originally came from https://github.com/technobium/weka-decision-trees

## Bayesian network

A Bayesian network is designed to predict an outcome on conditions.
Means if you can say how likely something happen, given a certain condition,
then you can combine the different conditions.

Variables are nodes and edges are conditional probabilities

![example](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/SimpleBayesNet.svg/800px-SimpleBayesNet.svg.png "example")

The math for conditional probability is simple, i will skip the implementation here.
Details are here https://en.wikipedia.org/wiki/Bayesian_network

## neural network

The core element behing a neural network is a perceptron.
A perceptron takes several binary inputs, x1,x2,… and produces a single binary output:


![perceptron](https://raw.githubusercontent.com/michaelgruczel/machine-learning-tutorial/master/images/perceptron.png "perceptron")

The inputs have different importance/impact, means they can be formalized as weights

![perceptron](https://raw.githubusercontent.com/michaelgruczel/machine-learning-tutorial/master/images/formular.png "formular")

Instead of a threshold a sigmoid function can be used as well (activation function)
This perceptrons can be layered as well.

![perceptron](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/500px-Colored_neural_network.svg.png "perceptron")

Such a neural network can learn to predict/classify more or less everything in theory.			
In order to so this, the perfect weights must be learned. 
There are several algorithms in place to simulate the learning.

A commonly used cost is the mean-squared error, which tries to minimize the average squared error between the network's output, 
f(x), and the target value y over all the example pairs. 

The following is a stochastic gradient descent algorithm for training a three-layer network (only one hidden layer):

    initialize network weights (often small random values)
    do
     forEach training example named ex
        prediction = neural-net-output(network, ex) 
        actual = teacher-output(ex)
        compute error (prediction - actual) at the output units
        compute delta for all weights from hidden layer to output layer
        compute delta for all weights input layer to hidden layer
        update network weights // input layer not modified by error estimate
      until all examples classified correctly or another stopping criterion satisfied
    return the network
  
You can find an example calculation with weka in WekaNeuralNetworkExampleApplication

## association rules learning

A method for discovering interesting relations between variables in large databases.
Often used in e-commerce to predict an output on base of the occurrence of another event.

details see https://en.wikipedia.org/wiki/Association_rule_learning

mahout is a tool which is often used. A good example can be found under https://chimpler.wordpress.com/2013/05/02/finding-association-rules-with-mahout-frequent-pattern-mining/
Here is a copy of that example included.
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

## clustering

The idea of clustering is to group items together. This is interesting for example for Market research or Crime analysis.

One important algorithm is K-Means Clustering:

* Step 1: setup randomly some clusters/groups, a cluster is point which represents a centroid
* Step 2: assign each object to the cluster/group that is nearest (has the closest centroid).
* Step 3: When all objects have been assigned, recalculate the positions of the centroids.
* Repeat Steps 2 and 3 until the centroids no longer move. 