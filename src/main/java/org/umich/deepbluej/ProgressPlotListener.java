package org.umich.deepbluej;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import javax.swing.JProgressBar;

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.style.Styler.ChartTheme;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.scijava.log.LogService;

import ij.gui.Plot;
import ij.gui.PlotWindow;

public class ProgressPlotListener
implements TrainingListener, EarlyStoppingListener {
	// Progress Monitors for GUI interface
	private JProgressBar epochProgress = null;
	private JProgressBar batchProgress = null;

	// Containers for plotting values
	private List<Double> epochInd = new ArrayList<Double>();
	private List<Double> trainScore = new ArrayList<Double>();
	private List<Double> testScore = new ArrayList<Double>();
	
	// Charting objects
	private XYChart progressChart;
	private SwingWrapper<XYChart> chartPanel;
	
	private long trainIterCount = 0;
	private long totalTrainIter = 0;
	private int numEpochs;
	private final Evaluation eval;
	private LogService log = null;
	private boolean invoked = false;
	private int numClasses;
	private int logFrequency = 0;
	private Plot plot;
	private PlotWindow plotWin;
	
	public ProgressPlotListener (int numClasses) {
		this.numClasses = numClasses;
		eval = new Evaluation(numClasses);
		trainScore.add((double) 0);
		epochInd.add((double) 1);
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
		double score = (trainScore.get(trainScore.size()-1) * (double) trainIterCount + model.score()) / ((double) trainIterCount + 1);
		trainScore.set(trainScore.size()-1, score);
		trainIterCount++;
		if (log!=null && trainIterCount%logFrequency==0) {
			log.info("Train: Epoch " + epochInd.get(epochInd.size()-1) + " (iteration " + trainIterCount + "/" + totalTrainIter + ") Objective: " + score);
		}
		if (batchProgress!=null) {
			batchProgress.setValue((int) trainIterCount);
			batchProgress.setString("Batch: " + Long.toString(trainIterCount) + "/" + Long.toString(totalTrainIter));
		}
	}

	@Override
	public void onStart(EarlyStoppingConfiguration esConfig, Model net) {
		if (epochProgress!=null) {
			int currentEpoch = epochInd.get(epochInd.size()-1).intValue();
			epochProgress.setValue(currentEpoch);
			epochProgress.setString("Epoch: " + Integer.toString(currentEpoch) + "/" + Integer.toString(numEpochs));
		}
	}

	@Override
	public void onEpoch(int epochNum, double score, EarlyStoppingConfiguration esConfig, Model net) {
		testScore.add(score);
		processEpoch(trainIterCount);
		if (epochProgress!=null) {
			int currentEpoch = epochInd.get(epochInd.size()-1).intValue();
			epochProgress.setValue(currentEpoch);
			epochProgress.setString("Epoch: " + Integer.toString(currentEpoch) + "/" + Long.toString(numEpochs));
		}
		if (batchProgress!=null) {
			batchProgress.setValue((int) trainIterCount);
			batchProgress.setString("Batch: " + Long.toString(trainIterCount) + "/" + Long.toString(totalTrainIter));
		}
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
		
		if (progressChart==null) {
			progressChart = new XYChart(800,600,ChartTheme.GGPlot2);
			progressChart.setTitle("Training Progress");
			progressChart.setXAxisTitle("Epochs");
			progressChart.setYAxisTitle("Objective");
			progressChart.addSeries("train", epochs, train);
			progressChart.addSeries("test", epochs, test);
			chartPanel = new SwingWrapper<XYChart>(progressChart);
			chartPanel.displayChart();
		} else {
			progressChart.updateXYSeries("train", epochs, train, null);
			progressChart.updateXYSeries("test", epochs, test,null);
			chartPanel.repaintChart();
		}
		
		if (numEpochs>epochInd.size()) {
			epochInd.add(epochInd.get(epochInd.size()-1)+1);
			trainScore.add((double) 0);
			trainIterCount = 0;			
		}
	}

	@Override
	public void onCompletion(EarlyStoppingResult esResult) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void onEpochStart(Model model) {

	}

	@Override
	public void onEpochEnd(Model model) {
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
	
	public void setNumBatches(int numBatches) {
		this.totalTrainIter = numBatches;
	}
	
	public void setNumEpochs(int numEpochs) {
		this.numEpochs = numEpochs;
	}
	
	public void setEpochPMon(JProgressBar epochProgress) {
		this.epochProgress = epochProgress;
	}
	
	public void setBatchPMon(JProgressBar batchProgress) {
		this.batchProgress = batchProgress;
	}

}
