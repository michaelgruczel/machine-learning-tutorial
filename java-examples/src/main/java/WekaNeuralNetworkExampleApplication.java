import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

import java.io.FileReader;

/**
 * Created by michaelgruczel on 26/11/16.
 */
public class WekaNeuralNetworkExampleApplication {

    public static void main(String[] args) throws Exception {
        try{
            // train
            FileReader trainreader = new FileReader("./../data/WekaNeuralNetworkExampleTitanic.arff");
            Instances train = new Instances(trainreader);
            train.setClassIndex(train.numAttributes()-1);
            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.setLearningRate(0.1);
            mlp.setMomentum(0.2);
            mlp.setTrainingTime(2000);
            mlp.setHiddenLayers("3");
            mlp.buildClassifier(train);

            // check quality of training
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(mlp, train);
            System.out.println(eval.errorRate()); //Printing Training Mean root squared Error
            System.out.println(eval.toSummaryString()); //Summary of Training


        }
        catch(Exception ex){
            ex.printStackTrace();
        }




    }
}
