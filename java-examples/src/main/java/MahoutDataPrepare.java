import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;

public class MahoutDataPrepare {

    // This program creates two files:
    // output.dat: contains the transaction data in the new format
    // mapping.csv: contains the mapping between the item name and the item id
    public static void main(String args[]) throws Exception {

        BufferedReader csvReader = new BufferedReader(new FileReader("./../data/MahoutAssociationLearningExample.csv"));

        // with the first line we can get the mapping id, text
        String line = csvReader.readLine();
        String[] tokens = line.split(",");
        FileWriter mappingWriter = new FileWriter("./../data/MahoutAssociationLearningExampleMapping.csv");
        int i = 0;
        for(String token: tokens) {
            mappingWriter.write(token.trim() + "," + i + "\n");
            i++;
        }
        mappingWriter.close();

        FileWriter datWriter = new FileWriter("./../data/MahoutAssociationLearningExampleOutput.dat");
        int transactionCount = 0;
        while(true) {
            line = csvReader.readLine();
            if (line == null) {
                break;
            }

            tokens = line.split(",");
            i = 0;
            boolean isFirstElement = true;
            for(String token: tokens) {
                if (token.trim().equals("true")) {
                    if (isFirstElement) {
                        isFirstElement = false;
                    } else {
                        datWriter.append(",");
                    }
                    datWriter.append(Integer.toString(i));
                }
                i++;
            }
            datWriter.append("\n");
            transactionCount++;
        }
        datWriter.close();
        System.out.println("Wrote " + transactionCount + " transactions.");
    }

}
