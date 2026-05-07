import java.util.Random;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

class ClusteringDataGeneration{

	public static double[][] generatePointsInRectangle(int count, double x1Min, double x1Max, double x2Min, double x2Max){
		double[][] points = new double[count][2];
		Random rand = new Random();
		
		for(int i=0;i<count;i++){
			points[i][0] = x1Min + rand.nextDouble() * (x1Max - x1Min);
			points[i][1] = x2Min + rand.nextDouble() * (x2Max - x2Min);
		}
		return points;
	}
	
	 public static double[][] generateAllGroups() {
        double[][] group1 = generatePointsInRectangle(100, -2, -1.6, 1.6, 2);
        double[][] group2 = generatePointsInRectangle(100, -1.2, -0.8, 1.6, 2);
        double[][] group3 = generatePointsInRectangle(100, -0.4, 0, 1.6, 2);
        double[][] group4 = generatePointsInRectangle(100, -1.8, -1.4, 0.8, 1.2);
        double[][] group5 = generatePointsInRectangle(100, -0.6, -0.2, 0.8, 1.2);
        double[][] group6 = generatePointsInRectangle(100, -2, -1.6, 0, 0.4);
        double[][] group7 = generatePointsInRectangle(100, -1.2, -0.8, 0, 0.4);
        double[][] group8 = generatePointsInRectangle(100, -0.4, 0, 0, 0.4);
        double[][] group9 = generatePointsInRectangle(200, -2, 0, 0, 2);

        double[][] totalPoints = new double[1000][3];
        int index = 0;
		double[][][] allGroups = {group1, group2, group3, group4, group5, group6, group7, group8, group9};
        for (double[][] group : allGroups) {
            for (double[] point : group) {
                totalPoints[index++] = point;
            }
        }

        return totalPoints;
    }
	
	public static void saveDataSetToFile(String filename, double[][] points){
		try(BufferedWriter writer = new BufferedWriter(new FileWriter(filename))){
			for (double[] point : points) {
                writer.write(String.format("%.3f %.3f%n", point[0], point[1]));
            }
		}
		catch(IOException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
	}
	
	public static void main(String args[]){
		double[][] totalPoints = generateAllGroups();
		saveDataSetToFile("clustering_data.txt", totalPoints);
	}

}