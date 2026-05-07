import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.io.BufferedWriter;
import java.io.FileWriter;

class PO {
    
    private double[][] points = new double[1000][2];
    private int M = 8; //number of clusters
    private double centroids[][];
	
	public double[][] getCentroids(){
		return centroids;
	}

    public void loadDataSet(String filename){
		
		List<double[]> pointData = new ArrayList<>();
        try (FileInputStream input = new FileInputStream(filename);
             Scanner scanner = new Scanner(input)) {

            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                String[] parts = line.split("\\s+"); 
                if (parts.length == 2) {
                    double x1 = Double.parseDouble(parts[0].replace(",", "."));
                    double x2 = Double.parseDouble(parts[1].replace(",", "."));
                    pointData.add(new double[]{x1, x2});
                }
            }
        } catch (IOException e) {
            System.out.println("An error occured\n" + e.getMessage());
        }

		for (int i = 0; i < pointData.size(); i++) {
			points[i] = pointData.get(i);
		}
    }

    public void kmeans(){
        setStartingCentroids();
        boolean centroidsChanged = true;

        while (centroidsChanged){

            //put each point in the correct cluster
            List<List<double[]>> clusters = new ArrayList<>();
            for (int i = 0; i < M; i++) {
                clusters.add(new ArrayList<>());
            }
            for (int i=0; i<points.length; i++){
                int nearestCluster = findNearestCluster(points[i]);
                clusters.get(nearestCluster).add(points[i]);
            }

            //update the centroids
            centroidsChanged = false;
            for (int i=0; i<centroids.length; i++){
                double[] newCentroid = calculateNewCentroid(clusters.get(i));
                
                if (!(centroids[i][0] == newCentroid[0] && centroids[i][1] == newCentroid[1])){
                    centroids[i][0] = newCentroid[0];
                    centroids[i][1] = newCentroid[1];
                    centroidsChanged = true;
                }
            }
        }
    }

    private double[] calculateNewCentroid(List<double[]> cluster){
        double[] newCentroid = new double[2];
        //iterating each point in cluster 
        for (int i=0; i<cluster.size(); i++){
            for (int j=0; j<2; j++){
                newCentroid[j] += cluster.get(i)[j];
            }
        }
        newCentroid[0] = newCentroid[0]/cluster.size();
        newCentroid[1] = newCentroid[1]/cluster.size();

        return newCentroid;
    }

    private int findNearestCluster(double[] x){
        double minDist = calculateEuclidianDistance(x,centroids[0]);
        int nearestCluster = 0;
        for (int i=1; i<centroids.length; i++){
            double dist = calculateEuclidianDistance(x,centroids[i]);
            if (dist < minDist){
                minDist = dist;
                nearestCluster = i;
            }
        }
        return nearestCluster;
    }

    private double calculateEuclidianDistance(double[] x, double[] w){
        double dist = (double) Math.pow(x[0]-w[0],2) + (double) Math.pow(x[1]-w[1],2);
        return dist;
    }

    public void setStartingCentroids(){
		this.centroids= new double[M][2];
        Random rnd = new Random();
        List<Integer> selectedIndexes = new ArrayList<>();
        int i=0;
        while (selectedIndexes.size() < M){
            int index = rnd.nextInt(1000);
            if (!selectedIndexes.contains(index)){
                selectedIndexes.add(index);
                centroids[i][0] = points[index][0];
                centroids[i][1] = points[index][1];
                i++;
            }
        }
    }

    public double calculateClusteringError(int runCounter){
        double error = 0;
        for (int i=0; i<points.length; i++){
            int clusterIndex = findNearestCluster(points[i]);
            error += calculateEuclidianDistance(points[i],centroids[clusterIndex]);
        }
        System.out.println("Total clustering error for run " + runCounter + ": "+ error);
		return error;
    }
	
	public void saveCentroidsToFile(double[][] centroids){
		String fileName = "CentroidsCoordinatesForM-" + M + ".txt";
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
			for(int i=0; i<M; i++){
				double xCentroid = centroids[i][0];
				double yCentroid = centroids[i][1];
				
				writer.write(xCentroid + " " + yCentroid + "\n");
			}
		}
		catch (IOException e) {
			System.err.println("Error while writing to file: " + e.getMessage());
		}
	}

    public static void main(String[] args){
		ArrayList<double[][]> centroidsOfEachAttempt = new ArrayList<double[][]>();
		ArrayList<Double> errorInEachAttempt = new ArrayList<Double>();
		PO model = new PO();
		model.loadDataSet("clustering_data.txt");
		for(int i=0; i<20; i++){
			model.kmeans();
			double error = model.calculateClusteringError(i);
			centroidsOfEachAttempt.add(model.getCentroids());
			errorInEachAttempt.add(error);
		}
		//Find the minimum error and its corresponding centroids
		double minError = errorInEachAttempt.get(0);
		double[][] bestCentroids = centroidsOfEachAttempt.get(0);
		for (int i = 1; i < errorInEachAttempt.size(); i++) {
			if (errorInEachAttempt.get(i) < minError) {
				minError = errorInEachAttempt.get(i);
				bestCentroids = centroidsOfEachAttempt.get(i);
			}
		}
		System.out.println("\n Minimum clustering error " + minError);
		model.saveCentroidsToFile(bestCentroids);
		
    }

}
