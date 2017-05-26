package org.umich.ij.dbj;

import java.io.IOException;
import java.util.Arrays;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class DemoData {	
	static public String[] DemoDataSets = {"NONE",
								   		   "MNIST"};

	private static String demoDir = null;
	
	public static String getDataDir() {
		return DemoData.demoDir;
	}
	
	public static void setDataDir(String demoDir) {
		DemoData.demoDir = demoDir;
	}
	
	public static DataSetIterator getTrainData(String dataSet,NetworkParams netParams) {
		dataSet = dataSet.toUpperCase();
		if (Arrays.asList(DemoDataSets).contains(dataSet)) {
			switch (dataSet) {
				case "MNIST":
					try {
						return new MnistDataSetIterator(netParams.batchSize,true,netParams.seed);
					} catch (IOException e) {
						e.printStackTrace();
					}
				default:
					return null;
			}
		} else {
			return null;
		}
	}
	
	public static DataSetIterator getTestData(String dataSet,NetworkParams netParams) {
		dataSet = dataSet.toUpperCase();
		if (Arrays.asList(DemoDataSets).contains(dataSet)) {
			switch (dataSet) {
				case "MNIST":
					try {
						return new MnistDataSetIterator(netParams.batchSize,false,netParams.seed);
					} catch (IOException e) {
						e.printStackTrace();
					}
				default:
					return null;
			}
		} else {
			return null;
		}
	}
}