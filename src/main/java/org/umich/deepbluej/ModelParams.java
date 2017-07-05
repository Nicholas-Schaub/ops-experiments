package org.umich.deepbluej;

import java.util.Arrays;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;

public class ModelParams {
	public static final int IMAGE_CLASSIFICATION = 0;
	public static final int IMAGE_REGRESSION = 1;
	public static final int IMAGE_ATTRIBUTES = 2;
	public static final int PIXEL_CLASSIFICATION = 3;
	public static final int PIXEL_REGRESSION = 4;
	public static final int PIXEL_ATTRIBUTES = 5;
	public static final String[] MODEL_TYPE_STRING = {"Img Class",
													  "Img Reg",
													  "Img Att",
													  "Pix Class",
													  "Pix Reg",
												   	  "Pix Att"};
	
	public static final int SIMPLE_UNITS = 0;
	public static final int INCEPTION_UNITS = 1;
	public static final int REDUCED_SIMPLE_UNITS = 2;
	public static final int MINCEPTION_UNITS = 3;
	public static final String[] UNIT_TYPE_STRING = {"Simple",
													 "Inception",
													 "Red Simple",
													 "MInception"};
	
	public static final String[] UNIT_ACTIVATION_STRING = Arrays.stream(Activation.values()).map(Enum::name).toArray(String[]::new);

	// Model Input Parameters
	private int rowsIn;
	private int colsIn;
	private int attributesIn;
	
	// Model Output Parameters
	private int rowsOut;
	private int colsOut;
	private int attributesOut;
	
	// Model Structure
	private int modelType = 0;
	private String modelTypeString = MODEL_TYPE_STRING[0];
	private int unitType = 0;
	private String unitTypeString = UNIT_TYPE_STRING[0];
	private int[] unitRepeat = {0};
	private int unitComplexity = 5;
	private int[] unitScale = {1};
	private int modelDepth = 1;
	private int modelDepthMax = 1;
	
	// Methods that need to be implemented into the GUI
	private String unitActivation = UNIT_ACTIVATION_STRING[0];
	private int seed = 0;
	private boolean isTrained = false;
	public boolean useBatchNorm = false;
	public boolean useRegularization = true;
	public double dropoutRate = 0.5;
	public double learningRate = 0.001;
	public String updater = Updater.NESTEROVS.toString();
	public String optimizer = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT.toString();
	public double momentum = 0.9;
	
	public ModelParams(int mType, int rIn, int cIn,int outFeatures) {
		modelType(mType);
		rowsIn(rIn);
		colsIn(cIn);
		if (modelType<3) {
			rowsOut(1);
			colsOut(1);
		} else {
			rowsOut(rIn);
			colsOut(cIn);
		}
		attributesIn(1);
		attributesOut(outFeatures);
	}
	
	public void unitType(int uType) {
		if (uType>=0 && uType<UNIT_TYPE_STRING.length) {
			unitType = uType;
			unitTypeString = UNIT_TYPE_STRING[uType];
		}
	}
	public int unitType() {
		return unitType;
	}
	public void unitType(String uType) {
		for (int t=0; t<UNIT_TYPE_STRING.length; t++) {
			if (uType.matches(UNIT_TYPE_STRING[t])) {
				unitType = t;
				unitTypeString = UNIT_TYPE_STRING[t];
				return;
			}
		}
	}
	public String unitTypeString() {
		return unitTypeString;
	}
	
	public void modelType(int mType) {
		if (mType>=0 && mType<MODEL_TYPE_STRING.length) {
			modelType = mType;
			modelTypeString = MODEL_TYPE_STRING[mType];
		} else {
			throw new Error("Invalid model type.");
		}
	}
	public int modelType() {
		return modelType;
	}
	public void modelType(String mType) {
		for (int t=0; t<MODEL_TYPE_STRING.length; t++) {
			if (mType.matches(MODEL_TYPE_STRING[t])) {
				modelType = t;
				modelTypeString = MODEL_TYPE_STRING[t];
				return;
			}
		}
	}
	public String modelTypeString() {
		return modelTypeString;
	}
	
	public void rowsOut(int rOut) {
		if (modelType>=4 && rOut>0) {
			rowsOut = rOut;
		} else {
			rowsOut = 1;
		}
	}
	public int rowsOut() {
		return rowsOut;
	}
	
	public void colsOut(int cOut) {
		if (modelType>=4 && cOut>0) {
			colsOut = cOut;
		} else {
			colsOut = 1;
		}
	}
	public int colsOut() {
		return colsOut;
	}
	
	public void rowsIn(int rIn) {
		rowsIn = rIn;
		updateMaxDepth();
	}
	public int rowsIn() {
		return rowsIn;
	}
	
	public void colsIn(int cIn) {
		colsIn = cIn;
		updateMaxDepth();
	}
	public int colsIn() {
		return colsIn;
	}
	
	public void attributesIn(int aIn) {
		attributesIn = aIn;
	}
	public int attributesIn() {
		return attributesIn;
	}
	
	public void attributesOut(int nAttributes) {
		if (modelType!=0 || modelType!=3) {
			attributesOut = nAttributes;
		} else {
			attributesOut = 0;
		}
	}
	public int attributesOut() {
		return attributesOut;
	}
	
	private int log2(int n) {
		// returns largest integer power of 2 that divides n
		return Integer.numberOfTrailingZeros(n & -n)+1;
	}
	
	private void updateMaxDepth() {
		modelDepthMax = Math.min(log2(rowsIn), log2(colsIn));
		modelDepth(modelDepth);
	}
	
	public void modelDepth(int mDepth) {
		if (mDepth<=modelDepthMax) {
			modelDepth = mDepth;
		}
	}
	public int modelDepth() {
		return modelDepth;
	}
	
	public int modelScalesMax() {
		return modelDepthMax;
	}
	
	public void unitScale(int[] uScale) {
		if (uScale.length>=1 && uScale.length<=modelDepth) {
			unitScale = uScale;
		}
	}
	public int[] unitScale() {
		return unitScale;
	}
	public String unitScaleString() {
		String uScaleStr = Integer.toString(unitScale[0]);
		for (int i = 1; i < unitScale.length; i++) {
			uScaleStr += ", " + Integer.toString(unitScale[i]);
		}
		return uScaleStr;
	}
	public void unitScale(String uScale) {
		uScale = uScale.replace(" ", "");
		String[] uScaleSplit = uScale.split(",");
		if (uScaleSplit.length>1) {
			unitScale = new int[modelDepth];
		} else {
			unitScale = new int[1];
		}
		for (int i = 0; i < unitScale.length; i++) {
			int scale = 0;
			if (uScaleSplit.length>=modelDepth) {
				scale = Integer.parseInt(uScaleSplit[i]);
			} else if (uScaleSplit.length<modelDepth) {
				if (i<uScaleSplit.length) {
					scale = Integer.parseInt(uScaleSplit[i]);
				} else {
					scale = Integer.parseInt(uScaleSplit[uScaleSplit.length-1]);
				}
			}
			unitScale[i] = scale;
		}
	}
	
	public void unitComplexity(int uComplexity) {
		if (uComplexity>=3) {
			if (uComplexity<=16) {
				unitComplexity = uComplexity;
			} else {
				unitComplexity = 16;
			}
		} else {
			unitComplexity = 3;
		}
	}
	public int unitComplexity() {
		return unitComplexity;
	}
	
	public void unitRepeat(int[] uRepeat) {
		if (uRepeat.length>=1 && uRepeat.length<=modelDepth) {
			unitRepeat = uRepeat;
		}
	}
	public int[] unitRepeat() {
		return unitRepeat;
	}
	public String unitRepeatString() {
		String uRepeatStr = Integer.toString(unitRepeat[0]);
		for (int i = 1; i < unitRepeat.length; i++) {
			uRepeatStr += ", " + Integer.toString(unitRepeat[i]);
		}
		return uRepeatStr;
	}
	public void unitRepeat(String uRepeat) {
		uRepeat = uRepeat.replace(" ", "");
		String[] uRepeatSplit = uRepeat.split(",");
		unitRepeat = new int[uRepeatSplit.length];
		if (uRepeatSplit.length>1) {
			unitRepeat = new int[modelDepth];
		} else {
			unitRepeat = new int[1];
		}
		for (int i = 0; i < unitRepeat.length; i++) {
			int scale = 0;
			if (uRepeatSplit.length>=modelDepth) {
				scale = Integer.parseInt(uRepeatSplit[i]);
			} else if (uRepeatSplit.length<modelDepth) {
				if (i<uRepeatSplit.length) {
					scale = Integer.parseInt(uRepeatSplit[i]);
				} else {
					scale = Integer.parseInt(uRepeatSplit[uRepeatSplit.length-1]);
				}
			}
			unitRepeat[i] = scale;
		}
	}
	
	public int seed() {
		return seed;
	}
	public void seed(int s) {
		this.seed = s;
	}
	
	public ComputationGraph getModel() {
		return UnitBuilder.getModelFromParams(this);
	}
}