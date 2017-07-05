package org.umich.deepbluej.ij.plugin;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JTextField;
import javax.swing.JToggleButton;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.termination.IterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.scijava.app.StatusService;
import org.scijava.command.CommandService;
import org.scijava.log.LogService;
import org.scijava.thread.ThreadService;
import org.scijava.ui.UIService;
import org.umich.deepbluej.ModelParams;
import org.umich.deepbluej.ProgressPlotListener;
import org.umich.deepbluej.TrainingParams;
import org.umich.ij.guitools.DirectoryChooserPanel;
import org.umich.ij.guitools.ValidatedTextField;
import org.umich.ij.guitools.ValidatorInt;

import ij.IJ;
import ij.ImagePlus;
import net.imagej.ops.OpService;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imglib2.IterableInterval;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

@SuppressWarnings("serial")
public class DeepBlueJGUI extends JFrame {
	
	// SciJava/ImageJ parameters
	private OpService ops;
	private LogService log;
	private StatusService status;
	private CommandService cmd;
	private ThreadService thread;
	private UIService ui;

	// Network Training Panel
	private JPanel trainingPanel;
	private JLabel trainEpochsLabel;
	private ValidatedTextField<Integer> trainEpochs;
	private JLabel trainBatchSizeLabel;
	private ValidatedTextField<Integer> trainBatchSize;
	private JLabel trainSeedLabel;
	private ValidatedTextField<Integer> trainSeed;
	private JButton showTestImage;
	private JButton startTraining;
	private JButton stopTraining;
	private JProgressBar epochProgress;
	private JProgressBar batchProgress;
	
	// DL4J Settings
	private String currentDataSet;
	private TrainingParams trainingParams = new TrainingParams();
	private ModelParams modelParams = new ModelParams(0,28,28,10);
	
	// Training Interrupter
	private TrainInterrupt trainInterrupt = new TrainInterrupt();
	
	public DeepBlueJGUI() {
		
		DeepBlueJGUI.setDefaultLookAndFeelDecorated(true);
		
		this.setTitle("DeepBlueJ - Alpha");
		this.setSize(new Dimension(451,501));
		this.setLayout(new GridBagLayout());
		
		initElements();
		drawElements();
		initListeners();
	}
	
	private void initElements() {
		try {
			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		} catch (ClassNotFoundException | InstantiationException | IllegalAccessException
				| UnsupportedLookAndFeelException e) {
			e.printStackTrace();
		}
			
		trainingPanel = new JPanel(new GridBagLayout());
		trainingPanel.setBorder(BorderFactory.createTitledBorder("Training Settings"));
			trainEpochsLabel = new JLabel("# of Epochs: ");
			trainEpochs = new ValidatedTextField<Integer>(9, Integer.toString(trainingParams.numEpochs),new ValidatorInt(1,1000000));
			trainBatchSizeLabel = new JLabel("Batch Size: ");
			trainBatchSize = new ValidatedTextField<Integer>(9, Integer.toString(trainingParams.batchSize),new ValidatorInt(1,512));
			trainSeedLabel = new JLabel("Random Seed: ");
			trainSeed = new ValidatedTextField<Integer>(9, Integer.toString(trainingParams.seed),new ValidatorInt(Integer.MIN_VALUE,Integer.MAX_VALUE));
			showTestImage = new JButton("Show Test Image");
			showTestImage.setToolTipText("Display a random test image.");
			showTestImage.setFocusPainted(false);
			showTestImage.setFocusable(false);
			startTraining = new JButton("Start Training");
			startTraining.setFocusPainted(false);
			startTraining.setFocusable(false);
			startTraining.setForeground(new Color(0,171,103));
			epochProgress = new JProgressBar(0,trainingParams.numEpochs);
			epochProgress.setValue(0);
			epochProgress.setString("-/-");
			epochProgress.setStringPainted(true);
			batchProgress = new JProgressBar(0,trainingParams.numEpochs);
			batchProgress.setValue(0);
			batchProgress.setString("-/-");
			batchProgress.setStringPainted(true);
			stopTraining = new JButton("Stop Training");
			stopTraining.setFocusPainted(false);
			stopTraining.setFocusable(false);
			stopTraining.setForeground(new Color(203,43,37));
	}
	
	private void drawElements() {
		// Setup the constraints
		GridBagConstraints c = new GridBagConstraints();
		c.insets = new Insets(1,1,1,1);
		c.anchor = GridBagConstraints.NORTH;
		c.fill = GridBagConstraints.BOTH;
		c.ipady = 0;
		c.ipadx = 0;
		c.gridwidth = 2;
		c.gridheight = 1;
		c.weightx = 1;
		c.weighty = 1;
		c.gridx = 0;
		c.gridy = 0;
		
		//
		// Training Panel
		//
		c.gridy = 2;
		c.gridwidth = 2;
		c.gridx = 0;
		c.ipadx = 0;
		c.fill = GridBagConstraints.BOTH;
		c.anchor = GridBagConstraints.NORTH;
		this.add(trainingPanel,c);
		
		c.fill = GridBagConstraints.NONE;
		c.gridy = 0;
		c.gridwidth = 1;
		c.anchor = GridBagConstraints.EAST;
		trainingPanel.add(trainEpochsLabel, c);
		c.gridx++;
		c.ipadx = 35;
		c.anchor = GridBagConstraints.WEST;
		trainingPanel.add(trainEpochs, c);
		c.gridx++;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.EAST;
		trainingPanel.add(trainBatchSizeLabel,c);
		c.gridx++;
		c.ipadx = 35;
		c.anchor = GridBagConstraints.WEST;
		trainingPanel.add(trainBatchSize, c);
		c.gridx++;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.EAST;
		trainingPanel.add(trainSeedLabel,c);
		c.gridx++;
		c.ipadx = 35;
		c.anchor = GridBagConstraints.WEST;
		trainingPanel.add(trainSeed, c);
		
		c.gridy++;
		c.gridx = 0;
		c.gridwidth = 6;
		c.fill = GridBagConstraints.HORIZONTAL;
		c.anchor = GridBagConstraints.CENTER;
		trainingPanel.add(epochProgress, c);
		c.gridy++;
		trainingPanel.add(batchProgress,c);
		
		c.gridy++;
		c.ipadx = 10;
		c.ipady = 10;
		c.gridx = 1;
		c.gridwidth = 2;
		c.fill = GridBagConstraints.NONE;
		c.anchor = GridBagConstraints.EAST;
		trainingPanel.add(startTraining, c);
		c.gridx+=2;
		c.anchor = GridBagConstraints.WEST;
		trainingPanel.add(stopTraining,c);
		
		this.setLocationRelativeTo(null);
	}
	
	private void initListeners() {
		
		stopTraining.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				trainInterrupt.interrupt();
			}
			
		});
		

		
		startTraining.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent arg0) {
				//trainModel();
				thread.run(() -> {trainModel();});
			}			
			
		});
		
	}
	
	private void trainModel() {
		if (trainData==null || testData==null) {
			IJ.error("No data is loaded");
		}
		
		log.info("Setting training parameters...");
		updateTrainingParams();
		
        log.info("Build model...");
		ComputationGraph model = modelParams.getModel();
        model.init();

        String tempDir = System.getProperty("user.home");
        EarlyStoppingModelSaver<ComputationGraph> saver = new LocalFileGraphSaver(tempDir);
        EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(trainingParams.numEpochs))
                .evaluateEveryNEpochs(1)
                .iterationTerminationConditions(trainInterrupt)
                .scoreCalculator(new DataSetLossCalculatorCG(testData, true))     //Calculate test set score
                .modelSaver(saver)
                .build();
        
        EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConf,model,trainData);
        
		// Set up the Progress listener
        ProgressPlotListener listener = new ProgressPlotListener(10);
        int batches = (int) Math.ceil((double) trainingParams.numTrain/(double) trainingParams.batchSize);
        listener.setNumBatches(batches);
        batchProgress.setMaximum(batches);
        listener.setBatchPMon(batchProgress);
        epochProgress.setMaximum(trainingParams.numEpochs);
        listener.setEpochPMon(epochProgress);
        listener.setNumEpochs(trainingParams.numEpochs);

        model.setListeners(listener);
        trainer.setListener(listener);
        
        trainer.fit();
	}
	
	private void updateTrainingParams() {
		trainingParams.batchSize = Integer.parseInt(trainBatchSize.getText());
		trainingParams.seed = Integer.parseInt(trainSeed.getText());
		trainingParams.numEpochs = Integer.parseInt(trainEpochs.getText());
		trainData = DemoData.getTrainData(currentDataSet, trainingParams);
		testData = DemoData.getTestData(currentDataSet, trainingParams);
	}
	
	private class TrainInterrupt implements IterationTerminationCondition {
		private boolean guiInterrupt = false;
		
		@Override
		public void initialize() {
			guiInterrupt = false;
		}
		
		public void interrupt() {
			log.info("Training interruption requested...");
			guiInterrupt = true;
		}

		@Override
		public boolean terminate(double lastMiniBatchScore) {
			if (guiInterrupt) {
				log.info("Sending interruption request...");
			}
			return guiInterrupt;
		}
		
	}

	public void setOps(OpService ops) {
		this.ops = ops;
	}
	
	public OpService getOps() {
		return ops;
	}
	
	public void setLog(LogService log) {
		this.log = log;
	}
	
	public LogService getLog() {
		return log;
	}
	
	public void setStatus(StatusService status) {
		this.status = status;
	}
	
	public StatusService getStatus() {
		return status;
	}
	
	public void setCommand(CommandService cmd) {
		this.cmd = cmd;
	}
	
	public CommandService getCommand() {
		return cmd;
	}
	
	public void setThread(ThreadService thread) {
		this.thread = thread;
	}
	
	public ThreadService getThread() {
		return thread;
	}
	
	public void setUi(UIService ui) {
		this.ui = ui;
	}
	
	public UIService getUi() {
		return ui;
	}
}
