import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.trees.Id3;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Weka decision trees demo.
 *
 */
public class WekaDecisionTreeApplication {

	private Instances trainingData;

	public static void main(String[] args) throws Exception {

        WekaDecisionTreeApplication decisionTree = new WekaDecisionTreeApplication("./../data/WekaDecisionTreeExampleTitanic.arff");
		Id3 id3tree = decisionTree.trainTheTree();

		// Print the resulted tree
		System.out.println(id3tree.toString());

		// Test the tree
		Instance testInstance = decisionTree.prepareTestInstance();
		int result = (int) id3tree.classifyInstance(testInstance);

		String readableResult = decisionTree.trainingData.attribute(3).value(result);
		System.out.println(" ----------------------------------------- ");
		System.out.println("Test data               : " + testInstance);
		System.out.println("Test data classification: " + readableResult);
	}

	public WekaDecisionTreeApplication(String fileName) {
		BufferedReader reader = null;
		try {
			// Read the training data
			reader = new BufferedReader(new FileReader(fileName));
			trainingData = new Instances(reader);

			// Setting class attribute
			trainingData.setClassIndex(trainingData.numAttributes() - 1);
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	private Id3 trainTheTree() {
		Id3 id3tree = new Id3();

		String[] options = new String[1];
		// Use unpruned tree.
		options[0] = "-U";

		try {
			id3tree.setOptions(options);
			id3tree.buildClassifier(trainingData);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return id3tree;
	}

	private Instance prepareTestInstance() {
		Instance instance = new Instance(3);
		instance.setDataset(trainingData);

		instance.setValue(trainingData.attribute(0), "male");
		instance.setValue(trainingData.attribute(1), "no");
		instance.setValue(trainingData.attribute(2), "yes");

		return instance;
	}
}