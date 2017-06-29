package org.umich.ij.dbj;

import java.util.Arrays;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

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
	public static final String[] UNIT_TYPE_STRING = {"Simple",
													 "Inception"};
	
	public static final String[] UNIT_ACTIVATION_STRING = Arrays.stream(Activation.values()).map(Enum::name).toArray(String[]::new);

	// Model Input Parameters
	private int rowsIn;
	private int colsIn;
	private int attributesIn;
	
	// Model Output Parameters
	private int rowsOut;
	private int colsOut;
	private int attributesOut;
	private int numClasses;
	
	// Model Structure
	private int modelType = 0;
	private String modelTypeString = MODEL_TYPE_STRING[0];
	private int unitType = 0;
	private String unitTypeString = UNIT_TYPE_STRING[0];
	private int[] unitRepeat = {0};
	private int unitComplexity = 5;
	private int[] unitScale = {1};
	private String unitActivation = UNIT_ACTIVATION_STRING[0];
	private int modelDepth = 1;
	private int modelDepthMax = 1;
	
	// Methods that need to be implemented into the GUI
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
		numClasses(outFeatures);
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
	
	public void numClasses(int nClasses) {
		if (modelType==0 || modelType==3) {
			numClasses = nClasses;
		} else {
			numClasses = 0;
		}
	}
	public int numClasses() {
		return numClasses;
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
	
	public ComputationGraph getModel() {
		System.out.println(unitTypeString);
		
		GraphBuilder model = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.iterations(1)
				.learningRate(0.01)
		   	    .weightInit(WeightInit.XAVIER)
				.updater(Updater.NESTEROVS).momentum(0.9)
				.regularization(true).l2(0.0005)
				.graphBuilder();
		model.setBackprop(true);
		model.setPretrain(false);
		model.addInputs("input");
		String inLayer = "input";
		String outLayer = "";
		for (int scale = 0; scale < modelDepth; scale++) {
			int repeats = 0;
			if (unitRepeat.length==1) {
				repeats = unitRepeat[0];
			} else {
				repeats = unitRepeat[scale];
			}
			switch (unitType) {
				case SIMPLE_UNITS:
					outLayer = createSimpleUnit(model,scale,repeats,inLayer);
					break;
				case INCEPTION_UNITS:
					outLayer = createInceptionUnit(model,scale,repeats,inLayer);
					break;
			}
			if (scale!=modelDepth-1) {
				inLayer = "u" + Integer.toString(scale) + "_m";
				maxLayer(model,inLayer,outLayer,2,2);
				outLayer = inLayer;
				System.out.println(inLayer + " -> " + outLayer);
			}
		}
		inLayer = outLayer;
		outLayer = "dense";
		int nOut = (int) Math.pow(2, modelDepth+unitComplexity);
		model.addLayer(outLayer,
					   new DenseLayer.Builder()
					   	   .activation(Activation.RELU)
					   	   //.nIn(nIn)
					   	   .nOut(nOut)
					   	   .build(),
				   	   inLayer);
		System.out.println(inLayer + " -> " + outLayer);
		inLayer = outLayer;
		outLayer = "classification";
		nOut = attributesOut;
		if (modelType==IMAGE_CLASSIFICATION || modelType==PIXEL_CLASSIFICATION) {
			nOut = numClasses;
		}
		model.addLayer(outLayer,
					   new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
					   	   //.nIn(nIn)
					   	   .nOut(nOut)
					   	   .activation(Activation.SOFTMAX)
					   	   .build(),
					   "dense");
		System.out.println(inLayer + " -> " + outLayer);
		model.setOutputs("classification");
		model.setInputTypes(InputType.convolutionalFlat(rowsIn, colsIn, attributesIn));
		return new ComputationGraph(model.build());
	}
	
	public String createInceptionUnit(GraphBuilder model, int unitNum, int unitRepeat,String inLayer) {
		String unitBase = "u" + Integer.toString(unitNum) + "_x";
		int uNum = Math.min(unitNum, unitScale.length-1);
		int totalOut = (int) Math.pow(2, unitComplexity+unitNum);
		
		for (int i = 0; i < unitRepeat+1; i++) {
			// All inception units must have at least 2 layers: a 1x1 filter and a max -> 1x1 filter
			String[] mergeLayers = new String[unitScale[unitNum]];
			// 1x1 filter
			mergeLayers[0] = unitBase + Integer.toString(i) + "_s0_c";
			convLayer(model,inLayer,mergeLayers[0],0,totalOut/2);
			// max -> 1x1 filter
			String dScale = unitBase + Integer.toString(i) + "_s0_m";
			maxLayer(model,inLayer,dScale,3,1);
			mergeLayers[1]= unitBase + Integer.toString(i) + "_s0_d";
			convLayer(model,dScale,mergeLayers[0],0,totalOut/(int) Math.pow(2, unitScale[unitNum]));
					
			for (int j = 1; j < unitScale[unitNum]; j++) {
				dScale = unitBase + Integer.toString(i) + "_s0_m";
			}

			inLayer = outLayer;
		}
		return inLayer;
	}
	
	public String createSimpleUnit(GraphBuilder model, int unitNum, int unitRepeat, String inLayer) {
		String unitBase = "u" + Integer.toString(unitNum) + "_x";
		int uNum = Math.min(unitNum, unitScale.length-1);
		int nOut = (int) Math.pow(2, unitComplexity+unitNum);
		for (int i = 0; i < unitRepeat+1; i++) {
			String outLayer = unitBase + Integer.toString(i) + "_c";
			convLayer(model,inLayer,outLayer,unitScale[uNum],nOut);
			inLayer = outLayer;
		}
		return inLayer;
	}
	
	private void maxLayer(GraphBuilder model, String inLayer, String outLayer, int scale, int stride) {
		model.addLayer(inLayer,
					   new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
					   	   .kernelSize(scale,scale)
					   	   .stride(stride,stride)
					   	   .build(),
				   	   outLayer);
		System.out.println(inLayer + " -> " + outLayer);
	}
	
	private void convLayer(GraphBuilder model, String inLayer, String outLayer, int scale, int numOut) {
		model.addLayer(outLayer,
					   new ConvolutionLayer.Builder(2*scale+1,2*scale+1)
					   	   .padding(new int[] {scale,scale})
					   	   .nOut(numOut)
					   	   .stride(1,1)
					   	   .activation(Activation.RELU)
					   	   .build(),
					   	inLayer);
		System.out.println(inLayer + " -> " + outLayer);
	}
}