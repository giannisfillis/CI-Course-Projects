import java.util.Random;

class Neuron{

	private float incomingWeights[];
	private float output;
	private float bias;
	private float derivativeDelta[];
	private float derivativeDeltaForBias;
	
	public Neuron(){
		
	}
	
	public Neuron(int previousLayerSize){
		this.incomingWeights= new float[previousLayerSize];
		initializeWeightsAndBias();
		derivativeDelta = new float[previousLayerSize];
		derivativeDeltaForBias = 0f;
		for(int i=0; i<previousLayerSize; i++){
			derivativeDelta[i] = 0f;
		}
	}
	
	public void cleanDerivativeDelta(){
		for(int i=0; i<derivativeDelta.length; i++){
			derivativeDelta[i] = 0f;
		}
		derivativeDeltaForBias = 0f;
	}
	
	public int getNumberOfDerivativeDelta(){
		return derivativeDelta.length;
	}
	
	public float getDerivativeDeltaForBias(){
		return derivativeDeltaForBias;
	}
	
	public void setDerivativeDeltaForBias(float derivativeDeltaForBias){
		this.derivativeDeltaForBias = derivativeDeltaForBias;
	}
	
	public float getSpecificDerivativeDelta(int index){
		return derivativeDelta[index];
	}
	
	public void setDerivativeDelta(int index,float newWeight){
		this.derivativeDelta[index]= newWeight;
	}
	
	public float[] getWeights(){
		return incomingWeights;
	}
	
	public float getSpecificWeight(int index){
		return incomingWeights[index];
	}
	
	public void setWeights(float[] weights){
		this.incomingWeights=weights;
	}
	
	public int getNumberOfWeights(){
		return incomingWeights.length;
	}
	
	public float getOutput(){
		return output;
	}
	
	public void setOutput(float output){
		this.output=output;
	}
	
	public float getBias(){
		return bias;
	}
	
	public void setBias(float bias){
		this.bias=bias;
	}
	
	public void setWeight(int index,float newWeight){
		this.incomingWeights[index]= newWeight;
	}
	
	 public void initializeWeightsAndBias() {
        Random random = new Random();

        for (int i = 0; i < incomingWeights.length; i++) {
            incomingWeights[i] = -1 + 2 * random.nextFloat();
        }
        bias = -1 + 2 * random.nextFloat();
    }
}