package org.umich.ij.dbj;

import java.util.List;
import java.util.Map;

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.scijava.log.LogService;

public class ProgressPlotListener
implements TrainingListener, EarlyStoppingListener {

	private long epochInd = 1;
	private long trainIterCount = 0;
	private double trainScore = 0;
	private double trainAccuracy = 0;
	private long totalTrainIter = 0;
	private double testScore = 0;
	private double testAccuracy = 0;
	private final Evaluation eval;
	private LogService log;
	private boolean invoked = false;
	private int numClasses;
	private int logFrequency = 0;
	
	public ProgressPlotListener (int numClasses) {
		this.numClasses = numClasses;
		eval = new Evaluation(numClasses);
	}
	
	public ProgressPlotListener(int numClasses, LogService log, int logFrequency) {
		this(numClasses);
		this.log = log;
		this.logFrequency = logFrequency;
	}

	@Override
	public boolean invoked() {
		return false;
	}

	@Override
	public void invoke() {
		// TODO Auto-generated method stub
		this.invoked = true;
	}

	public void iterationDone(Model model, int iteration) {
		trainScore = (trainScore * (double) trainIterCount + model.score()) / ((double) trainIterCount + 1);
		trainIterCount++;
		if (log!=null && trainIterCount%logFrequency==0) {
			log.info("Train: Epoch " + epochInd + " (iteration " + trainIterCount + "/" + totalTrainIter + ") Objective: " + trainScore);
		}
	}

	@Override
	public void onStart(EarlyStoppingConfiguration esConfig, Model net) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void onEpoch(int epochNum, double score, EarlyStoppingConfiguration esConfig, Model net) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void onCompletion(EarlyStoppingResult esResult) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void onEpochStart(Model model) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void onEpochEnd(Model model) {
		totalTrainIter = trainIterCount;
		if (log!=null) {
			log.info("Train: Epoch " + epochInd + " (iteration " + trainIterCount + "/" + totalTrainIter + ") Objective: " + trainScore);
		}
		epochInd += 1;
		trainIterCount = 0;
	}

	@Override
	public void onForwardPass(Model model, List<INDArray> activations) {

	}

	@Override
	public void onForwardPass(Model model, Map<String, INDArray> activations) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void onGradientCalculation(Model model) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void onBackwardPass(Model model) {
		// TODO Auto-generated method stub
		
	}

}
