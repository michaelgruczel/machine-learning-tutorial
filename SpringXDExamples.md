# Spring XD

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
