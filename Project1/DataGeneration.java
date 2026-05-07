import java.util.Random;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

class DataGeneration{
	
	
	
	public static float[][] generatePoints(int dataSetSize){
		float[][] points = new float[dataSetSize][2];
		Random rand = new Random();
		
		for(int i=0;i<dataSetSize;i++){
			points[i][0] = -1 + rand.nextFloat(2); // x1
            points[i][1] = -1 + rand.nextFloat(2); // x2
		}
		return points;
	}
	
	public static String[] classifyPoints(float[][] points){
		String[] labels = new String[points.length];
		for (int i = 0; i < points.length; i++) {
            float x1 = points[i][0];
            float x2 = points[i][1];
			
			if(Math.pow(x1 - 0.5, 2) + Math.pow(x2 - 0.5, 2) < 0.2){
				if(x2 > 0.5){
					labels[i] = "C1";
				}
				if(x2<0.5){
					labels[i] = "C2";
				}
			}
			else if(Math.pow(x1 + 0.5, 2) + Math.pow(x2 + 0.5, 2) < 0.2){
				if(x2>-0.5){
					labels[i] = "C1";
				}
				if(x2<-0.5){
					labels[i] = "C2";
				}
			}
			else if(Math.pow(x1 - 0.5, 2) + Math.pow(x2 + 0.5, 2) < 0.2){
				if(x2>-0.5){
					labels[i] = "C1";
				}
				if(x2<-0.5){
					labels[i] = "C2";
				}
			}
			else if(Math.pow(x1 + 0.5, 2) + Math.pow(x2 - 0.5, 2) < 0.2){
				if(x2>0.5){
					labels[i] = "C1";
				}
				if(x2<0.5){
					labels[i] = "C2";
				}
			}
			else if(x1*x2 >0){
				labels[i] = "C3";
			}
			else if(x1*x2<0){
				labels[i] = "C4";
			}
		}
		return labels;
	}
	
	public static void saveDataSetToFile(String filename, float[][] points, String[] labels){
		int counter = 0;
		try(BufferedWriter writer = new BufferedWriter(new FileWriter(filename))){
			for (int i = 0; i < points.length; i++) {
				writer.write(String.format("%.3f %.3f %s%n", points[i][0], points[i][1], labels[i]));
				counter++;
			}
		}
		catch(IOException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
		System.out.printf("counter is %d \n",counter);
	}
	
	public static void main(String[] args){
	//	int totalPoints = 8000;
		int trainingSetSize = 4000;
		int testSetSize = 4000;
		
		float[][] trainingSet = generatePoints(trainingSetSize);
		float[][] testSet = generatePoints(testSetSize);
		
		String[] trainingLabels = classifyPoints(trainingSet);
		String[] testLabels = classifyPoints(testSet);
		
		saveDataSetToFile("training_set.txt",trainingSet,trainingLabels);
		saveDataSetToFile("test_set.txt",testSet,testLabels);
	}
}