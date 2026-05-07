class Layer{
	
	private Neuron neurons[];
	private int numberOfNeurons;
	private int previousLayerSize;
	
	private String activationFunction;
	
	public Layer(int numberOfNeurons,int previousLayerSize,String activationFunction){
		this.numberOfNeurons= numberOfNeurons;
		this.previousLayerSize = previousLayerSize;
		neurons = new Neuron[numberOfNeurons];
		createNeurons(numberOfNeurons);
		this.activationFunction = activationFunction;
	}
	
	public void cleanDeltasOfNeurons(){
		for(int i=0; i<neurons.length; i++){
			neurons[i].cleanDerivativeDelta();
		}
	}
	
	public void createNeurons(int numberOfNeurons){
		for(int i=0;i<numberOfNeurons;i++){
			neurons[i]= new Neuron(previousLayerSize);
		}
	}
	
	public Neuron getNeuron(int index){
		return neurons[index];
	}
	
	public int getNumberOfNeurons(){
		return numberOfNeurons;
	}
	
	public String getActivationFunction(){
		return activationFunction;
	}
}