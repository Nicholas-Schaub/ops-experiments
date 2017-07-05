package org.umich.deepbluej;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class TrainingParams {
	// Training parameters
	public int batchSize = 128;
	public int seed = 1;
	public int numEpochs = 5;
	
	// Basic information about training data
	public String dataName = "None";
	public int numTrain = 0;
	public int numTest = 0;
	private int rowsIn = 0;
	private int colsIn = 0;
	private int attributesOut;
	
	// Training data and information
	private DataSetIterator trainData;
	private DataSetIterator testData;
	
	public DataSetIterator trainData() {
		return trainData;
	}
	public void trainData(DataSetIterator trainData) {
		this.numTrain = trainData.numExamples();
		this.trainData = trainData;
	}
	
	public DataSetIterator testData() {
		return testData;
	}
	public void testData(DataSetIterator testData) {
		this.numTest = testData.numExamples();
		this.testData = testData;
	}
	
	public int attributesOut() {
		return attributesOut;
	}
	public void attributesOut(int attOut) {
		this.attributesOut = attOut;
	}
	
	public int rowsIn() {
		return rowsIn;
	}
	public void rowsIn(int rIn) {
		this.rowsIn = rIn;
	}
	
	public int colsIn() {
		return colsIn;
	}
	public void colsIn(int cIn) {
		this.colsIn = cIn;
	}
}
