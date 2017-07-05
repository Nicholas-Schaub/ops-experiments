package org.umich.deepbluej;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class UnitBuilder {
	
	public static ComputationGraph getModelFromParams(ModelParams modelParams) {
		System.out.println(modelParams.unitTypeString());
		
		GraphBuilder model = new NeuralNetConfiguration.Builder()
				.seed(modelParams.seed())
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
		for (int depth = 0; depth < modelParams.modelDepth(); depth++) {
			int repeats = 0;
			if (modelParams.unitRepeat().length==1) {
				repeats = modelParams.unitRepeat()[0];
			} else {
				repeats = modelParams.unitRepeat()[depth];
			}
			int scale = 0;
			if (modelParams.unitRepeat().length==1) {
				scale = modelParams.unitScale()[0];
			} else {
				scale = modelParams.unitScale()[depth];
			}
			switch (modelParams.unitType()) {
				case ModelParams.SIMPLE_UNITS:
					outLayer = createSimpleUnit(model,depth,scale,modelParams.unitComplexity()+depth,repeats,inLayer);
					break;
				case ModelParams.INCEPTION_UNITS:
					outLayer = createInceptionUnit(model,depth,scale,modelParams.unitComplexity()+depth,repeats,inLayer);
					break;
				case ModelParams.REDUCED_SIMPLE_UNITS:
					outLayer = createSimpleLRUnit(model,depth,scale,modelParams.unitComplexity()+depth,repeats,inLayer);
					break;
				case ModelParams.MINCEPTION_UNITS:
					outLayer = createMInceptionUnit(model,depth,scale,modelParams.unitComplexity()+depth,repeats,inLayer);
					break;
			}
			if (scale!=modelParams.modelDepth()-1) {
				inLayer = outLayer;
				outLayer = "u" + Integer.toString(scale) + "_m";
				maxLayer(model,inLayer,outLayer,2,2,0);
				inLayer = outLayer;
			}
		}
		inLayer = outLayer;
		outLayer = "dense";
		int nOut = (int) Math.pow(2, modelParams.modelDepth()+modelParams.unitComplexity());
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
		nOut = modelParams.attributesOut();
		model.addLayer(outLayer,
					   new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
					   	   .nOut(nOut)
					   	   .activation(Activation.SOFTMAX)
					   	   .build(),
					   "dense");
		System.out.println(inLayer + " -> " + outLayer);
		model.setOutputs("classification");
		model.setInputTypes(InputType.convolutionalFlat(modelParams.rowsIn(), modelParams.colsIn(), modelParams.attributesIn()));
		return new ComputationGraph(model.build());
	}
	
	public static String createInceptionUnit(GraphBuilder model, int unitNum, int unitScale, int unitComplexity, int unitRepeat, String inLayer) {
		String unitBase = "u" + Integer.toString(unitNum) + "_x";
		int totalOut = (int) Math.pow(2, unitComplexity);
		
		for (int i = 0; i < unitRepeat+1; i++) {
			// All inception units must have at least 2 layers: a 1x1 filter and a max -> 1x1 filter
			String[] mergeLayers = new String[unitScale+2];
			// 1x1 filter
			mergeLayers[0] = unitBase + Integer.toString(i) + "_s0_c";
			convLayer(model,inLayer,mergeLayers[0],0,totalOut/2);
			// 3x3 max -> 1x1 filter
			String dScale = unitBase + Integer.toString(i) + "_s0_m";
			maxLayer(model,inLayer,dScale,3,1,1);
			mergeLayers[1]= unitBase + Integer.toString(i) + "_s0_d";
			convLayer(model,dScale,mergeLayers[1],0,totalOut/(int) Math.pow(2, unitScale + 1));
			
			// Create additional layers at different scales
			for (int j = 1; j < unitScale+1; j++) {
				dScale = unitBase + Integer.toString(i) + "_s" + Integer.toString(j) + "_d";
				convLayer(model,inLayer,dScale,0,totalOut/(int) Math.pow(2, j + 2));
				mergeLayers[j+1] = unitBase + Integer.toString(i) + "_s" + Integer.toString(j) + "_c";
				convLayer(model,dScale,mergeLayers[j+1],j,totalOut/(int) Math.pow(2, j + 1));
			}
			inLayer = unitBase + Integer.toString(i) + "_merge";
			for (int k = 0; k < mergeLayers.length-1; k++) {
				System.out.print(mergeLayers[k] + ", ");
			}
			System.out.print(mergeLayers[mergeLayers.length-1] + " -> " + inLayer);
			System.out.println();
			model.addVertex(inLayer, new MergeVertex(), mergeLayers);
		}
		return inLayer;
	}
	
	public static String createMInceptionUnit(GraphBuilder model, int unitNum, int unitScale, int unitComplexity, int unitRepeat,String inLayer) {
		String unitBase = "u" + Integer.toString(unitNum) + "_x";
		int totalOut = (int) Math.pow(2, unitComplexity+unitNum);
		
		for (int i = 0; i < unitRepeat+1; i++) {
			// All inception units must have at least 2 layers: a 1x1 filter and a max -> 1x1 filter
			String[] mergeLayers = new String[unitScale+2];
			// 1x1 filter
			mergeLayers[0] = unitBase + Integer.toString(i) + "_s0_c";
			convLayer(model,inLayer,mergeLayers[0],0,totalOut/2);
			// In the original inception model, a 3x3 max filter was applied followed by 1x1 filter.
			// To increase speed, reverse the order.
			// 1x1 filter -> 3x3 max
			String dScale = unitBase + Integer.toString(i) + "_s0_d";
			convLayer(model,inLayer,dScale,0,totalOut/(int) Math.pow(2, unitScale + 1));
			mergeLayers[1] = unitBase + Integer.toString(i) + "_s0_m";
			maxLayer(model,dScale,mergeLayers[1],3,1,1);
			
			// Create additional layers at different scales
			for (int j = 1; j < unitScale+1; j++) {
				dScale = unitBase + Integer.toString(i) + "_s" + Integer.toString(j) + "_d";
				int kappa = totalOut/(int) Math.pow(2, j + 2);
				convLayer(model,inLayer,dScale,0,kappa);
				mergeLayers[j+1] = unitBase + Integer.toString(i) + "_s" + Integer.toString(j) + "_c";
				convLRLayer(model,dScale,mergeLayers[j+1],j,kappa*2,kappa);
			}
			inLayer = unitBase + Integer.toString(i) + "_merge";
			for (int k = 0; k < mergeLayers.length-1; k++) {
				System.out.print(mergeLayers[k] + ", ");
			}
			System.out.print(mergeLayers[mergeLayers.length-1] + " -> " + inLayer);
			System.out.println();
			model.addVertex(inLayer, new MergeVertex(), mergeLayers);
		}
		return inLayer;
	}
	
	public static String createSimpleUnit(GraphBuilder model, int unitNum, int unitScale, int unitComplexity, int unitRepeat, String inLayer) {
		String unitBase = "u" + Integer.toString(unitNum) + "_x";
		int nOut = (int) Math.pow(2, unitComplexity);
		for (int i = 0; i < unitRepeat+1; i++) {
			String outLayer = unitBase + Integer.toString(i) + "_c";
			convLayer(model,inLayer,outLayer,unitScale,nOut);
			inLayer = outLayer;
		}
		return inLayer;
	}
	
	public static String createSimpleLRUnit(GraphBuilder model, int unitNum, int unitScale, int unitComplexity, int unitRepeat, String inLayer) {
		String unitBase = "u" + Integer.toString(unitNum) + "_x";
		int nOut = (int) Math.pow(2, unitComplexity);
		int kappa = (int) Math.max(8, nOut/2);
		for (int i = 0; i < unitRepeat+1; i++) {
			String outLayer = unitBase + Integer.toString(i) + "_c";
			convLRLayer(model,inLayer,outLayer,unitScale,nOut,kappa);
			inLayer = outLayer;
		}
		return inLayer;
	}
	
	private static void mergeLayer(GraphBuilder model, String[] mergeLayers, String outLayer) {
		
	}
	
	private static void maxLayer(GraphBuilder model, String inLayer, String outLayer, int scale, int stride, int pad) {
		model.addLayer(outLayer,
					   new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
					   	   .kernelSize(scale,scale)
					   	   .padding(new int[] {pad, pad})
					   	   .stride(stride,stride)
					   	   .build(),
				   	   inLayer);
		System.out.println(inLayer + " -> " + outLayer);
	}
	
	private static void convLayer(GraphBuilder model, String inLayer, String outLayer, int scale, int numOut) {
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
	
	private static void convLRLayer(GraphBuilder model, String inLayer, String outLayer, int scale, int numOut, int kappa) {
		String lrLayer = inLayer + "_lr";
		System.out.println(inLayer + " -> " + lrLayer);
		model.addLayer(lrLayer,
					   new ConvolutionLayer.Builder(2*scale+1,1)
					   	   .nOut(kappa)
					   	   .stride(1,1)
					   	   .activation(Activation.RELU)
					   	   .build(),
					   	inLayer);
		model.addLayer(outLayer,
				   new ConvolutionLayer.Builder(1,2*scale+1)
				   	   .padding(new int[] {scale,scale})
				   	   .nOut(numOut)
				   	   .stride(1,1)
				   	   .activation(Activation.RELU)
				   	   .build(),
				   	lrLayer);
		System.out.println(lrLayer + " -> " + outLayer);
	}
}
