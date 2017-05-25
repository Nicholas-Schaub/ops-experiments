package org.umich.ij.dbj;

public class NetworkParams {
	public String dataName = "None";
	
	// DL4J input parameters that need to be set.
	public int numRowsIn;
	public int numColsIn;
	public int numClasses;
	
	// DL4J input parameters with defaults
	public int batchSize = 128;
	public int seed = 1;
	public int numEpochs = 5;
	
	public NetworkParams() {
		
	}
}
