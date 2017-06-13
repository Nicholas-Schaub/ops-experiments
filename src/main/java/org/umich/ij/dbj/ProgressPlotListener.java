package org.umich.ij.dbj;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import javax.swing.SwingUtilities;

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

	// Containers for plotting values
	private List<Double> epochInd = new ArrayList<Double>();
	private List<Double> trainScore = new ArrayList<Double>();
	private List<Double> testScore = new ArrayList<Double>();
	
	// Charting objects
	private XYChart progressChart;
	private SwingWrapper<XYChart> chartPanel;
	
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
		epochInd.add((double) 1);
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
		log.info("Creating double arrays...");
		log.info("epochs.length: " + epochs.length);
		log.info("trainScore.length: " + train.length);
		log.info("test.length: " + test.length);
		for (int i = 0; i < epochs.length; i++) {
			log.info(i);
			log.info("epochInd");
			epochs[i] = (double) epochInd.get(i);
			log.info("trainScore");
			train[i] = (double) trainScore.get(i);
			log.info("testScore");
			test[i] = (double) testScore.get(i);
			log.info("Done");
		}
		
		log.info("Drawing train scores...");
		if (progressChart==null) {
			log.info("XYChart...");
			progressChart = new XYChart(800,600,ChartTheme.GGPlot2);
			log.info("XYChart Title...");
			progressChart.setTitle("Training Progress");
			log.info("XYChart Epochs...");
			progressChart.setXAxisTitle("Epochs");
			log.info("XYChart Objective...");
			progressChart.setYAxisTitle("Objective");
			log.info("XYChart TrainData...");
			progressChart.addSeries("train", epochs, train);
			log.info("XYChart TestData...");
			progressChart.addSeries("test", epochs, test);
			log.info("invokeLater...");
			//SwingUtilities.invokeLater(() -> {
			log.info("Creating swing wrapper...");
			chartPanel = new SwingWrapper<XYChart>(progressChart);
			log.info("Displaying chart...");
			chartPanel.displayChart();
//			});
//			JFrame frame = new JFrame("Training Progress");
//			frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//			frame.add(chartPanel);
//			frame.setVisible(true);
//			Time.sleep(500);
		} else {
			log.info("UpdateXYSeries...");
			progressChart.updateXYSeries("train", epochs, train, null);
			progressChart.updateXYSeries("test", epochs, test,null);
			log.info("Repaint...");
			chartPanel.repaintChart();
//			Time.sleep(500);
		}
		epochInd.add(epochInd.get(epochInd.size()-1)+1);
		trainScore.add((double) 0);
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
