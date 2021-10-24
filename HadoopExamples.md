# hadoop

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
