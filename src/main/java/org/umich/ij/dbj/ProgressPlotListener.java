package org.umich.ij.dbj;

import java.util.ArrayList;
import java.util.Collections;
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

import ij.gui.Plot;
import ij.gui.PlotWindow;

public class ProgressPlotListener
implements TrainingListener, EarlyStoppingListener {

	// Containers for plotting values
	private ArrayList<Long> epochInd = new ArrayList<Long>();
	private ArrayList<Double> trainScore = new ArrayList<Double>();
	private ArrayList<Double> testScore = new ArrayList<Double>();
	
	
	private long trainIterCount = 0;
	private long totalTrainIter = 0;
	private final Evaluation eval;
	private LogService log;
	private boolean invoked = false;
	private int numClasses;
	private int logFrequency = 0;
	private Plot plot;
	private PlotWindow plotWin;
	
	public ProgressPlotListener (int numClasses) {
		this.numClasses = numClasses;
		eval = new Evaluation(numClasses);
	}
	
	public ProgressPlotListener(int numClasses, LogService log, int logFrequency) {
		this(numClasses);
		this.log = log;
		this.logFrequency = logFrequency;
		trainScore.add((double) 0);
		testScore.add((double) 0);
		epochInd.add((long) 1);
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
		double score = (trainScore.get(trainScore.size()-1) * (double) trainIterCount + model.score()) / ((double) trainIterCount + 1);
		trainScore.set(trainScore.size()-1, score);
		trainIterCount++;
		if (log!=null && trainIterCount%logFrequency==0) {
			log.info("Train: Epoch " + epochInd.get(epochInd.size()-1) + " (iteration " + trainIterCount + "/" + totalTrainIter + ") Objective: " + score);
		}
	}

	@Override
	public void onStart(EarlyStoppingConfiguration esConfig, Model net) {

	}

	@Override
	public void onEpoch(int epochNum, double score, EarlyStoppingConfiguration esConfig, Model net) {
		log.info("onEpoch");
		testScore.add(score);
		processEpoch(trainIterCount);
	}
	
	private void processEpoch(long trainIter) {
		totalTrainIter = trainIterCount;
		if (testScore.get(testScore.size()-1)>0 && log!=null) {
			log.info("Test: Epoch " + epochInd.get(epochInd.size()-1) + " Objective: " + testScore.get(testScore.size()-1));
		}
		
		double[] epochs = new double[epochInd.size()];
		double[] train = new double[trainScore.size()];
		double[] test = new double[testScore.size()];
		for (int i = 0; i < epochs.length; i++) {
			epochs[i] = (double) epochInd.get(i);
			train[i] = (double) trainScore.get(i);
			test[i] = (double) testScore.get(i);
		}
		
		log.info("Drawing train scores...");
		plot = new Plot("Training Progress","Epoch","Objective",epochs,train,Plot.CROSS);
		plot.setLimits(epochInd.get(0)-1, epochInd.get(epochInd.size()-1)+1, Collections.min(trainScore)*1.1, Collections.max(trainScore)*0.9);
		//log.info("Drawing train scores...");
		//plot.setColor(Color.BLACK);
		//plot.addPoints(epochs, train,Plot.LINE);
		plot.draw();
		log.info("Drawing test scores...");
		//plot.setColor(Color.RED);
		plot.addPoints(epochs, test,Plot.CROSS);
		if (plotWin==null) {
			plotWin = plot.show();
		} else {
			plotWin.drawPlot(plot);
		}
		
		epochInd.add(epochInd.get(epochInd.size()-1)+1);
		trainIterCount = 0;
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
		log.info("onEpochEnd");
		processEpoch(trainIterCount);
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
