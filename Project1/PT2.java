
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.io.BufferedWriter;
import java.io.FileWriter;


class PT2{
	public static final int d = 2; //input
	public static final int K = 4; // num of categories
	public static final int H1 = 3; // neurons in first hidden layer
	public static final int H2 = 6; // neurons in second hidden layer
	public static final String relu = "relu";
	public static final String tanh = "tanh";
	
	public static final String secondHiddenLayerActivationFunction = tanh;
	public static final String firstHiddenLayerActivationFunction = tanh;
	
	public float y[]; //output of MLP
	
	public static final int trainingSetSize = 4000;
	public static final int testSetSize = 4000;
	
	public float x[][] = new float[trainingSetSize][2]; // input
	public float t[][] = new float[trainingSetSize][4]; // desired output
	
	public float testX[][] = new float[testSetSize][2];
	public float testT[][] = new float[testSetSize][4];
	

	public int N = 4000;
	public int B = N/20;
	
	public float squaredErrorOfEpoch = 0;
	
	public float totalSquaredError = 0;
	
	public float threshold = 0.01f;
	
	float[] currentError = new float[1000*trainingSetSize];
	
	
	public Layer layers[] = new Layer[4];
	
	private float learningRate = 0.1f;
	
	public float gradientHidden1[][];
	public float gradientHidden2[][];
	public float gradientOutputLayer[][];
	
	public float gradientBiasHidden1[];
	public float gradientBiasHidden2[];
	public float gradientBiasOutputLayer[];
	
	public PT2(){
	}
	
	public void createModel(){
		layers[0] = new Layer(d,0,"none");
		layers[1] = new Layer(H1,d,firstHiddenLayerActivationFunction);
		layers[2]= new Layer(H2,H1,secondHiddenLayerActivationFunction);
		layers[3] = new Layer(K,H2,"logistic");
		
		gradientHidden1 = new float[layers[1].getNumberOfNeurons()][d];
		gradientHidden2 = new float[layers[2].getNumberOfNeurons()][layers[1].getNumberOfNeurons()];
		gradientOutputLayer = new float[K][layers[2].getNumberOfNeurons()] ;
		
		gradientBiasHidden1 = new float[layers[1].getNumberOfNeurons()];
		gradientBiasHidden2 = new float[layers[2].getNumberOfNeurons()];
		gradientBiasOutputLayer = new float[K];
		
		initializeWeightsOfLayers();
		System.out.println("weights initialised ");
	}
	
	public void loadTestSet(String filename){
		
		List<float[]> pointData = new ArrayList<>();
        List<String> labelData = new ArrayList<>();
        try (FileInputStream input = new FileInputStream(filename);
             Scanner scanner = new Scanner(input)) {

            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                String[] parts = line.split("\\s+"); 
                if (parts.length == 3) {
                    float x1 = Float.parseFloat(parts[0].replace(",", "."));
                    float x2 = Float.parseFloat(parts[1].replace(",", "."));
                    String label = parts[2];

                    pointData.add(new float[]{x1, x2});
                    labelData.add(label);
                }
            }
        } catch (IOException e) {
            System.out.println("An error occured\n" + e.getMessage());
        }

		for (int i = 0; i < pointData.size(); i++) {
			testX[i] = pointData.get(i);
		}
		for (int i = 0; i < labelData.size(); i++) {
			if(labelData.get(i).equals("C1")){
				testT[i] = new float[]{1,0,0,0};
			}
			else if(labelData.get(i).equals("C2")){
				testT[i] = new float[]{0,1,0,0};
			}
			else if(labelData.get(i).equals("C3")){
				testT[i] = new float[]{0,0,1,0};
			}
			else{
				testT[i] = new float[]{0,0,0,1};
			}
		}
	}
	
	public void loadDataSet(String filename){
		
		List<float[]> pointData = new ArrayList<>();
        List<String> labelData = new ArrayList<>();
        try (FileInputStream input = new FileInputStream(filename);
             Scanner scanner = new Scanner(input)) {

            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                String[] parts = line.split("\\s+"); 
                if (parts.length == 3) {
                    float x1 = Float.parseFloat(parts[0].replace(",", "."));
                    float x2 = Float.parseFloat(parts[1].replace(",", "."));
                    String label = parts[2];

                    pointData.add(new float[]{x1, x2});
                    labelData.add(label);
                }
            }
        } catch (IOException e) {
            System.out.println("An error occured\n" + e.getMessage());
        }

		for (int i = 0; i < pointData.size(); i++) {
			x[i] = pointData.get(i);
		}
		for (int i = 0; i < labelData.size(); i++) {
			if(labelData.get(i).equals("C1")){
				t[i] = new float[]{1,0,0,0};
			}
			else if(labelData.get(i).equals("C2")){
				t[i] = new float[]{0,1,0,0};
			}
			else if(labelData.get(i).equals("C3")){
				t[i] = new float[]{0,0,1,0};
			}
			else{ 
				t[i] = new float[]{0,0,0,1};
			}
		}
	}
	
	public float activate(float sum,String activationFunction){
		float result = 0;
		if(activationFunction.equals("tanh")){
			result= (float) Math.tanh(sum);
		}else if(activationFunction.equals("logistic")){
			result = 1.0f / (1.0f + (float) Math.exp(-sum));
		}
		else{ // relu
			if(sum > 0){
				result = sum;
			}
			else{
				result = 0;
			}
		}
		return result;
	}
	
	
	public float[] forward_pass(float x[], int d, float[] y, int K){
		float[] outputOfHiddenLayer1 = computeLayer(x,layers[1]);
		float[] outputOfHiddenLayer2 = computeLayer(outputOfHiddenLayer1,layers[2]);

		
		float[] outputOfOutputLayer = computeLayer(outputOfHiddenLayer2,layers[3]);
		y = outputOfOutputLayer;
		return outputOfOutputLayer;
	}
	
	public float[] computeLayer(float[] previousLayerOutput,Layer currentLayer){
		
		float[] outputOfTheLayer = new float[currentLayer.getNumberOfNeurons()];
		String activationFunctionOfLayer = currentLayer.getActivationFunction();
		int totalNeuronsInLayer = currentLayer.getNumberOfNeurons();
		
		for(int i=0;i<totalNeuronsInLayer;i++){ // for every neuron in the layer
			
			Neuron currentNeuron = currentLayer.getNeuron(i);
			int totalWeightsOfNeuron = currentNeuron.getNumberOfWeights();
			float sum = currentNeuron.getBias();
			
			for(int j=0;j<totalWeightsOfNeuron;j++){ // for each weight of the neuron
				sum += currentNeuron.getSpecificWeight(j) * previousLayerOutput[j];
			}
			
			currentNeuron.setOutput(activate(sum,activationFunctionOfLayer));
			outputOfTheLayer[i] = currentNeuron.getOutput();
		}
		return outputOfTheLayer;
	}
	
	public void backprop(float[] x, int d, float[] t, int K){
		float[] predictedOutput = forward_pass(x,d,y,K);
		float[] outputError = new float[K];
		
		// computing the error for the output layer
		for(int i=0;i<K;i++){ 
			outputError[i] = predictedOutput[i] - t[i]; 	
			squaredErrorOfEpoch += outputError[i]*outputError[i];
		}
		
		float[] outputDelta = new float[K];
		for(int i=0;i<K;i++){ 
			outputDelta[i] = outputError[i]*derivativeLogistic(predictedOutput[i]); 			
		}
		
		// computing the error for the second hidden layer
		float[] hidden2_delta = new float[layers[2].getNumberOfNeurons()];
		for(int i=0;i<layers[2].getNumberOfNeurons();i++){ // for each neuron in hidden layer 2
			
			for(int j=0;j<K;j++){ // for each neuron in output
				float weightFromHiddenToOutput = layers[3].getNeuron(j).getSpecificWeight(i);
				hidden2_delta[i] += outputDelta[j]*weightFromHiddenToOutput;
			}
			float hiddenNeuronOutputValue = layers[2].getNeuron(i).getOutput();
			if(layers[2].getActivationFunction().equals("tanh")){
				hidden2_delta[i] = hidden2_delta[i]*derivativeTanh(hiddenNeuronOutputValue);
			}else{ // activation function is relu
				hidden2_delta[i] = hidden2_delta[i]*derivativeRelu(hiddenNeuronOutputValue);
			}
		}
		
		// computing the error for the first hidden layer
		float[] hidden1_delta = new float[layers[1].getNumberOfNeurons()];
		for(int i=0;i<layers[1].getNumberOfNeurons();i++){ // for each neuron in hidden layer 1
						
			for(int j=0;j<layers[2].getNumberOfNeurons();j++){ // for each neuron in hidden layer 2 
				float weightFromHiddenOneToHiddenTwo = layers[2].getNeuron(j).getSpecificWeight(i);
				hidden1_delta[i] += hidden2_delta[j]*weightFromHiddenOneToHiddenTwo;
			}
			float hidden1NeuronOutputValue = layers[1].getNeuron(i).getOutput();
			if(layers[1].getActivationFunction().equals("tanh")){
				hidden1_delta[i] = hidden1_delta[i]*derivativeTanh(hidden1NeuronOutputValue);
			}
			else{
				hidden1_delta[i] = hidden1_delta[i]*derivativeRelu(hidden1NeuronOutputValue);
			}
		}
		
		
		// updating gradient from hidden layer 2 to output
		for(int i=0;i<K;i++){ // iterating through the output layer
			
			float gradient = 0;
			for(int j=0;j<layers[2].getNumberOfNeurons();j++){ //iterating through the second hidden layer
				float outputOfHiddenLayer2Neuron = layers[2].getNeuron(j).getOutput();
				gradient = outputDelta[i]*outputOfHiddenLayer2Neuron;
				gradientOutputLayer[i][j] = gradient;
				
			}
			float gradientForBias = outputDelta[i];
			gradientBiasOutputLayer[i] =  gradientForBias;
		}
		
		// updating delta derivatives from hidden layer 1 to hidden layer 2
		for(int i=0; i<layers[2].getNumberOfNeurons();i++){ //iterating through the second hidden layer
			
			float gradient = 0;
			for(int j=0;j<layers[1].getNumberOfNeurons();j++){ //iterating through the first hidden layer
				float outputOfHiddenLayer1Neuron = layers[1].getNeuron(j).getOutput();
				gradient = hidden2_delta[i]*outputOfHiddenLayer1Neuron;
				gradientHidden2[i][j] = gradient;
			}
			float gradientForBias = hidden2_delta[i];
			gradientBiasHidden2[i] = gradientForBias;
		}
		
		
		// updating delta derivatives from input layer to hidden layer 1
		for(int i=0; i<layers[1].getNumberOfNeurons();i++){ //iterating through the first hidden layer
			float gradient = 0;
			for(int j=0; j<d; j++){ //iterating through the input layer
				float outputOfInputLayer = x[j];
				gradient = hidden1_delta[i]*outputOfInputLayer;

				gradientHidden1[i][j] = gradient;
			}
			float gradientForBias = hidden1_delta[i];
			gradientBiasHidden1[i] = gradientForBias;
		} 
		
	}
	
	
	private float derivativeLogistic(float input){
		return input*(1-input);
	}
	
	private float derivativeTanh(float input){
		float tanhValue = (float) Math.tanh(input);
		return 1- tanhValue*tanhValue;
	}
	
	private float derivativeRelu(float input){
		float result = 0;
		if(input <= 0){
			result = 0; 
		}
		else{
			result = 1;
		}
		return result;
	}
	
	public void initializeWeightsOfLayers(){
		
		// first hidden layer
		for(int i=0; i<layers[1].getNumberOfNeurons();i++){
			Neuron neuron = layers[1].getNeuron(i);
			neuron.initializeWeightsAndBias();
		}
		
		// second hidden layer
		for(int i=0; i<layers[2].getNumberOfNeurons();i++){
			Neuron neuron = layers[2].getNeuron(i);
			neuron.initializeWeightsAndBias();
		}
		
		// output hidden layer
		for(int i=0; i<layers[3].getNumberOfNeurons();i++){
			Neuron neuron = layers[3].getNeuron(i);
			neuron.initializeWeightsAndBias();
		}
	}
	
	public void gradientDescent(){
		int batchIteration = 1;
		int epochs = 0;
		while (conditionEvaluation(epochs)){
			squaredErrorOfEpoch = 0; // initializing again the squared error
			
			for(int i=0; i<trainingSetSize; i++){
				backprop(x[i], d, t[i], K);
				updateGradients(layers[1],1); // first hidden layer
				updateGradients(layers[2],2); // second hidden layer
				updateGradients(layers[3],3); // output layer
				if (batchIteration == B){ // a batch has been completed
					updateWeightsOfLayer(layers[1]); // first hidden layer
					updateWeightsOfLayer(layers[2]); // second hidden layer
					updateWeightsOfLayer(layers[3]); // output layer
					resetGradientArrays();
					
					deleteDeltasOfLayers();
					batchIteration = 0;
				}
				batchIteration++;
			}
			float totalerror = calculateError();
			currentError[epochs] = squaredErrorOfEpoch/2;
			System.out.println("Epoch: "+epochs+" Total error: "+totalerror);
			epochs++;
		}
	}
	
	private boolean conditionEvaluation(int epochs){
		if(epochs<800){
			return true;
		}
		else{
			if(Math.abs(currentError[epochs-1]-currentError[epochs-2])>=threshold){
				return true;
			}
		}
		return false;
	}
	
	private void deleteDeltasOfLayers(){
		layers[1].cleanDeltasOfNeurons();
		layers[2].cleanDeltasOfNeurons();
		layers[3].cleanDeltasOfNeurons();
	}
	
	private void resetGradientArrays(){
		resetTempDerivatives(gradientHidden1);
		resetTempDerivatives(gradientHidden2);
		resetTempDerivatives(gradientOutputLayer);
		
		resetTempDerivativesBias(gradientBiasHidden1);
		resetTempDerivativesBias(gradientBiasHidden2);
		resetTempDerivativesBias(gradientBiasOutputLayer);
	}
	
	private void resetTempDerivatives(float[][] derivatives){
		for(int i=0; i<derivatives.length; i++){
			for(int j=0; j<derivatives[i].length; j++){
				derivatives[i][j] = 0f;
			}
		}
	}
	
	private void resetTempDerivativesBias(float[] derivativesBias){
		for(int i=0; i<derivativesBias.length; i++){
			derivativesBias[i] = 0f;
		}
	}
	
	private float calculateError(){
		totalSquaredError = squaredErrorOfEpoch/2; 
		return totalSquaredError;
	}
	private void updateWeightsOfLayer(Layer layer){
		
		for(int i=0; i<layer.getNumberOfNeurons();i++){
			Neuron neuron = layer.getNeuron(i);
			for(int j=0; j<neuron.getNumberOfWeights(); j++){
					float error = neuron.getSpecificDerivativeDelta(j);

					float newWeight = neuron.getSpecificWeight(j) - (learningRate*error);
					neuron.setWeight(j,newWeight);
	
			}
			float errorForBias = neuron.getDerivativeDeltaForBias();
			float newBias = neuron.getBias() - learningRate*errorForBias;
			neuron.setBias(newBias);
		}
	}
	
	private void updateGradients(Layer layer, int numberOfLayer){
		for(int i=0; i<layer.getNumberOfNeurons();i++){
			Neuron neuron = layer.getNeuron(i);
			for(int j=0; j<neuron.getNumberOfDerivativeDelta(); j++){
				if(numberOfLayer == 1){ // first hidden layer
					neuron.setDerivativeDelta(j, neuron.getSpecificDerivativeDelta(j) + gradientHidden1[i][j]);
				}
				if(numberOfLayer == 2){ // second hidden layer
					neuron.setDerivativeDelta(j, neuron.getSpecificDerivativeDelta(j) + gradientHidden2[i][j]);
				}
				if(numberOfLayer == 3){ // output layer
					neuron.setDerivativeDelta(j, neuron.getSpecificDerivativeDelta(j) + gradientOutputLayer[i][j]);
				}
			}
			if(numberOfLayer == 1){ // first hidden layer
					neuron.setDerivativeDeltaForBias( neuron.getDerivativeDeltaForBias() + gradientBiasHidden1[i]);
				}
			if(numberOfLayer == 2){ // second hidden layer
				neuron.setDerivativeDeltaForBias( neuron.getDerivativeDeltaForBias() + gradientBiasHidden2[i]);
			}
			if(numberOfLayer == 3){ // output layer
				neuron.setDerivativeDeltaForBias( neuron.getDerivativeDeltaForBias() + gradientBiasOutputLayer[i]);
			}
		}
	}
	
	public void testGeneralAbility(){
		String fileName = "PT2-" + H1 + "-" + H2 + "-" + B + "-" + layers[2].getActivationFunction() +".txt";
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
			int correctPredictions = 0;
			int testMade =0;
			String trueClass = " ";
			String predictedClass = " ";
			for(int i=0; i<testX.length; i++){
				float[] outputOfMLP = forward_pass(testX[i], d, testT[i], K);
				int labelOfOutput = findMaxIndex(outputOfMLP);
				int trueLabel = findMaxIndex(testT[i]);
				testMade++;
			
				if(labelOfOutput == trueLabel){
					correctPredictions++;
				}
				trueClass = convertToStringClass(trueLabel);
				predictedClass = convertToStringClass(labelOfOutput);
				float x1 = testX[i][0];
				float x2 = testX[i][1];
				writer.write(x1 + " " + x2 + " " + predictedClass + " " + trueClass + "\n");
			}
			float accuracy = (correctPredictions/( float) testSetSize)*100;
			System.out.println("The generalization ability is " + accuracy + " % ");
		}
		catch (IOException e) {
			System.err.println("Error while writing to file: " + e.getMessage());
		}
	}
	
	private String convertToStringClass(int label){
		String stringLabel = " ";
		if(label == 0){
			stringLabel = "C1"; 
		}
		else if(label == 1){
			stringLabel = "C2";
		}
		else if(label == 2){
			stringLabel = "C3";
		}else{
			stringLabel = "C4";
		}
		return stringLabel;
	}
	
	private int findMaxIndex(float[] labels){
		int maxIndex = 0;
		for(int i=1; i<labels.length; i++){
			if(labels[i] > labels[maxIndex]){
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	public static void main(String[] args){
		PT2 model = new PT2();
		
		model.loadDataSet("training_set.txt");
		model.loadTestSet("test_set.txt");

		model.createModel();
		model.gradientDescent();
		
		model.testGeneralAbility();

	}
}