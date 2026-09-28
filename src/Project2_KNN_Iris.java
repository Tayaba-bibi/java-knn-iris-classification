import java.io.*;
import java.util.*;

public class Project2_KNN_Iris {

    // =========================================================
    // IRIS DATA
    // =========================================================
    static class Iris {
        double[] features;
        int label;

        Iris(double[] features, int label) {
            this.features = features;
            this.label = label;
        }
    }

    // =========================================================
    // NEIGHBOR
    // =========================================================
    static class Neighbor {
        double distance;
        int label;

        Neighbor(double distance, int label) {
            this.distance = distance;
            this.label = label;
        }
    }

    // =========================================================
    // SCALER
    // =========================================================
    static class Scaler {

        double[] mean;
        double[] standardDeviation;

        Scaler(int numberOfFeatures) {
            mean = new double[numberOfFeatures];
            standardDeviation =
                    new double[numberOfFeatures];
        }

        // Learn mean and standard deviation
        // from training data only
        void fit(List<Iris> data) {

            int numberOfFeatures = mean.length;

            // Mean
            for (Iris flower : data) {
                for (int i = 0; i < numberOfFeatures; i++) {
                    mean[i] += flower.features[i];
                }
            }

            for (int i = 0; i < numberOfFeatures; i++) {
                mean[i] /= data.size();
            }

            // Standard deviation
            for (Iris flower : data) {
                for (int i = 0; i < numberOfFeatures; i++) {

                    double difference =
                            flower.features[i] - mean[i];

                    standardDeviation[i] +=
                            difference * difference;
                }
            }

            for (int i = 0; i < numberOfFeatures; i++) {

                standardDeviation[i] =
                        Math.sqrt(
                                standardDeviation[i]
                                        / data.size()
                        );

                if (standardDeviation[i] == 0) {
                    standardDeviation[i] = 1;
                }
            }
        }

        // Scale dataset
        void transform(List<Iris> data) {

            for (Iris flower : data) {

                for (int i = 0; i < mean.length; i++) {

                    flower.features[i] =
                            (flower.features[i] - mean[i])
                                    / standardDeviation[i];
                }
            }
        }

        // Scale a new flower
        double[] transformNewFlower(
                double[] originalFeatures) {

            double[] scaled =
                    new double[originalFeatures.length];

            for (int i = 0;
                 i < originalFeatures.length;
                 i++) {

                scaled[i] =
                        (originalFeatures[i] - mean[i])
                                / standardDeviation[i];
            }

            return scaled;
        }
    }

    // =========================================================
    // EUCLIDEAN DISTANCE
    // =========================================================
    static double distance(
            double[] a,
            double[] b) {

        double sum = 0;

        for (int i = 0; i < a.length; i++) {

            double difference =
                    a[i] - b[i];

            sum += difference * difference;
        }

        return Math.sqrt(sum);
    }

    // =========================================================
    // KNN PREDICTION
    // =========================================================
    static int predict(
            List<Iris> trainingData,
            double[] testFeatures,
            int k) {

        List<Neighbor> neighbors =
                new ArrayList<>();

        // Calculate distance from every training sample
        for (Iris flower : trainingData) {

            double d =
                    distance(
                            flower.features,
                            testFeatures
                    );

            neighbors.add(
                    new Neighbor(d, flower.label)
            );
        }

        // Nearest first
        neighbors.sort(
                Comparator.comparingDouble(
                        n -> n.distance
                )
        );

        // Count votes
        int[] votes = new int[3];

        for (int i = 0; i < k; i++) {
            votes[neighbors.get(i).label]++;
        }

        // Find class with highest votes
        int bestLabel = 0;

        for (int i = 1; i < 3; i++) {

            if (votes[i] > votes[bestLabel]) {
                bestLabel = i;
            }
        }

        return bestLabel;
    }

    // =========================================================
    // ACCURACY
    // =========================================================
    static double accuracy(
            List<Iris> data,
            List<Iris> trainingData,
            int k) {

        int correct = 0;

        for (Iris flower : data) {

            int prediction =
                    predict(
                            trainingData,
                            flower.features,
                            k
                    );

            if (prediction == flower.label) {
                correct++;
            }
        }

        return correct * 100.0 / data.size();
    }

    // =========================================================
    // 5-FOLD CROSS VALIDATION
    // =========================================================
    static double crossValidationAccuracy(
            List<Iris> data,
            int k,
            int folds) {

        List<Iris> shuffled =
                new ArrayList<>(data);

        Collections.shuffle(
                shuffled,
                new Random(100)
        );

        int foldSize =
                shuffled.size() / folds;

        int totalCorrect = 0;
        int totalSamples = 0;

        for (int fold = 0;
             fold < folds;
             fold++) {

            int start =
                    fold * foldSize;

            int end;

            if (fold == folds - 1) {
                end = shuffled.size();
            } else {
                end = start + foldSize;
            }

            List<Iris> validationData =
                    new ArrayList<>(
                            shuffled.subList(
                                    start,
                                    end
                            )
                    );

            List<Iris> foldTraining =
                    new ArrayList<>();

            for (int i = 0;
                 i < shuffled.size();
                 i++) {

                if (i < start || i >= end) {
                    foldTraining.add(
                            shuffled.get(i)
                    );
                }
            }

            for (Iris flower :
                    validationData) {

                int prediction =
                        predict(
                                foldTraining,
                                flower.features,
                                k
                        );

                if (prediction == flower.label) {
                    totalCorrect++;
                }

                totalSamples++;
            }
        }

        return totalCorrect * 100.0
                / totalSamples;
    }

    // =========================================================
    // FLOWER NAME
    // =========================================================
    static String flowerName(int label) {

        switch (label) {

            case 0:
                return "Setosa";

            case 1:
                return "Versicolor";

            default:
                return "Virginica";
        }
    }

    // =========================================================
    // MAIN
    // =========================================================
    public static void main(String[] args) {

        String filePath = "data/iris.csv";

        List<Iris> allData =
                new ArrayList<>();

        // =====================================================
        // 1. LOAD DATASET
        // =====================================================

        try {

            BufferedReader reader =
                    new BufferedReader(
                            new FileReader(filePath)
                    );

            // Skip header
            reader.readLine();

            String line;

            while ((line = reader.readLine()) != null) {

                String[] values =
                        line.split(",");

                if (values.length < 5) {
                    continue;
                }

                double[] features = {
                        Double.parseDouble(values[0]),
                        Double.parseDouble(values[1]),
                        Double.parseDouble(values[2]),
                        Double.parseDouble(values[3])
                };

                int label;

                switch (values[4].trim()) {

                    case "setosa":
                        label = 0;
                        break;

                    case "versicolor":
                        label = 1;
                        break;

                    case "virginica":
                        label = 2;
                        break;

                    default:
                        continue;
                }

                allData.add(
                        new Iris(features, label)
                );
            }

            reader.close();

        } catch (IOException e) {

            System.out.println(
                    "ERROR: iris.csv could not be found."
            );

            System.out.println(
                    "Check: Project2-KNN-Iris/data/iris.csv"
            );

            return;
        }

        // =====================================================
        // HEADER
        // =====================================================

        System.out.println();
        System.out.println(
                "=================================================="
        );
        System.out.println(
                "        IRIS FLOWER CLASSIFICATION"
        );
        System.out.println(
                "        K-NEAREST NEIGHBORS (KNN)"
        );
        System.out.println(
                "=================================================="
        );

        // =====================================================
        // 2. DATASET INFORMATION
        // =====================================================

        System.out.println();
        System.out.println("DATASET INFORMATION");
        System.out.println("-------------------");

        System.out.println(
                "Total samples : " + allData.size()
        );

        System.out.println(
                "Features      : 4"
        );

        System.out.println(
                "Classes       : 3"
        );

        System.out.println(
                "Classes       : Setosa, Versicolor, Virginica"
        );

        // =====================================================
        // 3. SHUFFLE
        // =====================================================

        Collections.shuffle(
                allData,
                new Random(42)
        );

        System.out.println();
        System.out.println(
                "Data shuffled successfully."
        );

        // =====================================================
        // 4. 80/20 TRAIN-TEST SPLIT
        // =====================================================

        int total = allData.size();

        int trainingSize =
                (int) (total * 0.80);

        List<Iris> trainingData =
                new ArrayList<>(
                        allData.subList(
                                0,
                                trainingSize
                        )
                );

        List<Iris> testData =
                new ArrayList<>(
                        allData.subList(
                                trainingSize,
                                total
                        )
                );

        System.out.println();
        System.out.println("DATA SPLIT");
        System.out.println("----------");

        System.out.println(
                "Training : "
                        + trainingData.size()
                        + " samples (80%)"
        );

        System.out.println(
                "Testing  : "
                        + testData.size()
                        + " samples (20%)"
        );

        // =====================================================
        // 5. STANDARDIZATION
        // =====================================================

        Scaler scaler =
                new Scaler(4);

        // Learn only from training data
        scaler.fit(trainingData);

        // Scale training data
        scaler.transform(trainingData);

        // Scale test data using training parameters
        scaler.transform(testData);

        System.out.println();
        System.out.println("FEATURE SCALING");
        System.out.println("---------------");

        System.out.println(
                "Standardization completed."
        );

        System.out.println(
                "Mean and standard deviation learned from training data."
        );

        System.out.println(
                "Same scaling applied to test data."
        );

        // =====================================================
        // 6. FIND BEST K
        // =====================================================

        int[] kValues = {
                1, 3, 5, 7, 9
        };

        int bestK = kValues[0];

        double bestAccuracy = -1;

        System.out.println();
        System.out.println("K VALUE COMPARISON");
        System.out.println("------------------");

        System.out.println(
                "5-Fold Cross Validation:"
        );

        for (int k : kValues) {

            double currentAccuracy =
                    crossValidationAccuracy(
                            trainingData,
                            k,
                            5
                    );

            System.out.printf(
                    "K = %d   Accuracy = %.2f%%%n",
                    k,
                    currentAccuracy
            );

            if (currentAccuracy > bestAccuracy) {

                bestAccuracy =
                        currentAccuracy;

                bestK = k;
            }
        }

        System.out.println();
        System.out.println(
                "Selected K = " + bestK
        );

        System.out.printf(
                "Best Cross-Validation Accuracy = %.2f%%%n",
                bestAccuracy
        );

        // =====================================================
        // 7. FINAL TEST
        // =====================================================

        double testAccuracy =
                accuracy(
                        testData,
                        trainingData,
                        bestK
                );

        System.out.println();
        System.out.println("FINAL TEST RESULT");
        System.out.println("-----------------");

        System.out.printf(
                "Test Accuracy = %.2f%%%n",
                testAccuracy
        );

        // =====================================================
        // 8. CONFUSION MATRIX
        // =====================================================

        int[][] confusionMatrix =
                new int[3][3];

        for (Iris flower : testData) {

            int prediction =
                    predict(
                            trainingData,
                            flower.features,
                            bestK
                    );

            confusionMatrix
                    [flower.label]
                    [prediction]++;
        }

        System.out.println();
        System.out.println("CONFUSION MATRIX");
        System.out.println("----------------");

        System.out.printf(
                "%-15s %-10s %-12s %-10s%n",
                "Actual",
                "Setosa",
                "Versicolor",
                "Virginica"
        );

        for (int actual = 0;
             actual < 3;
             actual++) {

            System.out.printf(
                    "%-15s",
                    flowerName(actual)
            );

            for (int predicted = 0;
                 predicted < 3;
                 predicted++) {

                System.out.printf(
                        "%-12d",
                        confusionMatrix
                                [actual]
                                [predicted]
                );
            }

            System.out.println();
        }

        // =====================================================
        // 9. PRECISION, RECALL, F1
        // =====================================================

        System.out.println();
        System.out.println(
                "EVALUATION METRICS"
        );
        System.out.println(
                "------------------"
        );

        System.out.printf(
                "%-12s %-12s %-12s %-12s%n",
                "Class",
                "Precision",
                "Recall",
                "F1"
        );

        for (int classIndex = 0;
             classIndex < 3;
             classIndex++) {

            int truePositive =
                    confusionMatrix
                            [classIndex]
                            [classIndex];

            int falsePositive = 0;
            int falseNegative = 0;

            for (int i = 0; i < 3; i++) {

                if (i != classIndex) {

                    falsePositive +=
                            confusionMatrix
                                    [i]
                                    [classIndex];

                    falseNegative +=
                            confusionMatrix
                                    [classIndex]
                                    [i];
                }
            }

            double precision =
                    truePositive + falsePositive == 0
                            ? 0
                            : (double) truePositive
                            / (truePositive
                            + falsePositive);

            double recall =
                    truePositive + falseNegative == 0
                            ? 0
                            : (double) truePositive
                            / (truePositive
                            + falseNegative);

            double f1 =
                    precision + recall == 0
                            ? 0
                            : 2 * precision * recall
                            / (precision + recall);

            System.out.printf(
                    "%-12s %-12.2f %-12.2f %-12.2f%n",
                    flowerName(classIndex),
                    precision,
                    recall,
                    f1
            );
        }

        // =====================================================
        // 10. NEW FLOWER PREDICTION
        // =====================================================

        double[] newFlower = {
                5.1,
                3.5,
                1.4,
                0.2
        };

        // Scale new flower using training-data parameters
        double[] scaledNewFlower =
                scaler.transformNewFlower(
                        newFlower
                );

        int prediction =
                predict(
                        trainingData,
                        scaledNewFlower,
                        bestK
                );

        System.out.println();
        System.out.println(
                "NEW FLOWER PREDICTION"
        );
        System.out.println(
                "---------------------"
        );

        System.out.println(
                "Sepal Length : " + newFlower[0]
        );

        System.out.println(
                "Sepal Width  : " + newFlower[1]
        );

        System.out.println(
                "Petal Length : " + newFlower[2]
        );

        System.out.println(
                "Petal Width  : " + newFlower[3]
        );

        System.out.println();
        System.out.println(
                "Predicted Class : "
                        + flowerName(prediction)
        );

        // =====================================================
        // COMPLETE
        // =====================================================

        System.out.println();
        System.out.println(
                "=================================================="
        );

        System.out.println(
                "             PROJECT COMPLETED"
        );

        System.out.println(
                "=================================================="
        );
    }
}