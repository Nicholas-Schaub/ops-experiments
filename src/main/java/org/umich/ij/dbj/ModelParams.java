package org.umich.ij.dbj;

public class ModelParams {
	public static final int IMAGE_CLASSIFICATION = 1;
	public static final int IMAGE_REGRESSION = 2;
	public static final int IMAGE_ATTRIBUTES = 3;
	public static final int PIXEL_CLASSIFICATION = 4;
	public static final int PIXEL_REGRESSION = 5;
	public static final int PIXEL_ATTRIBUTES = 6;
	
	public int modelType = 1;
	
	// Model Parameters
	public int numRowsIn;
	public int numColsIn;
	public int numClasses;
	
	public ModelParams() {
		
	}
}