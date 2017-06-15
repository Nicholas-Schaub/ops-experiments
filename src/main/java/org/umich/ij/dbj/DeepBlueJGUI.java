package org.umich.ij.dbj;

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
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JToggleButton;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.IterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.scijava.app.StatusService;
import org.scijava.command.CommandService;
import org.scijava.log.LogService;
import org.scijava.thread.ThreadService;
import org.scijava.ui.UIService;
import org.umich.ij.guitools.DirectoryChooserPanel;

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
public class DeepBlueJGUI extends JDialog{
	
	// SciJava/ImageJ parameters
	private OpService ops;
	private LogService log;
	private StatusService status;
	private CommandService cmd;
	private ThreadService thread;
	private UIService ui;
	
	// Demo data set load panel & components
	private JPanel demoDataPanel;
	private DirectoryChooserPanel dataDirectory;
	private JButton loadButton;
	private JToggleButton mnist;
	
	// Visualize data panel
	private JPanel visDataPanel;
	private JButton showAllClasses;

	// Network Training Panel
	private JLabel trainDataNameLabel;
	private JLabel trainDataName;
	private JLabel trainEpochsLabel;
	private JLabel trainEpochs;
	private JLabel trainBatchSizeLabel;
	private JLabel trainBatchSize;
	private JLabel trainSeedLabel;
	private JLabel trainSeed;
	private JPanel trainingPanel;
	private JLabel modelNumClassLabel;
	private JLabel modelNumClass;
	private JLabel modelInpWidthLabel;
	private JLabel modelInpWidth;
	private JLabel modelInpHeightLabel;
	private JLabel modelInpHeight;
	private JLabel modelTypeLabel;
	private JLabel modelType;
	private JButton startTraining;
	private JButton stopTraining;
	
	// DL4J Settings
	private String currentDataSet;
	private DataSetIterator trainData;
	private DataSetIterator testData;
	private TrainingParams trainingParams;
	private ModelParams modelParams;
	
	// Training Interrupter
	private TrainInterrupt trainInterrupt = new TrainInterrupt();
	
	public DeepBlueJGUI() {
		
		DeepBlueJGUI.setDefaultLookAndFeelDecorated(true);
		
		this.setTitle("DeepBlueJ - Alpha");
		this.setSize(new Dimension(501,501));
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
		
		// Create the demo panel
		//demoPanel = new JPanel(new GridBagLayout());
		
		// Create the download subpanel
		demoDataPanel = new JPanel(new GridBagLayout());
		demoDataPanel.setBorder(BorderFactory.createTitledBorder("Load Demo Data"));
			mnist = new JToggleButton("MNIST");
			mnist.setToolTipText("Load the MNIST digits dataset.");
			mnist.setFocusPainted(false);
			mnist.setFocusable(false);
			dataDirectory = new DirectoryChooserPanel("Save Directory: ",
													  ij.IJ.getDirectory("home") + "DeepBlueJ Demo Data" + File.separator,
													  20);
			dataDirectory.setToolTipText("Select a directory to download and unpack data to.");
			loadButton = new JButton("Load Data");
			loadButton.setFocusPainted(false);
			loadButton.setFocusable(false);
			
		visDataPanel = new JPanel(new GridBagLayout());
		visDataPanel.setBorder(BorderFactory.createTitledBorder("Visualize Data"));
			showAllClasses = new JButton("Show Each Class");
			showAllClasses.setToolTipText("Display a representative image for each class.");
			showAllClasses.setFocusPainted(false);
			showAllClasses.setFocusable(false);
		
		// Start and stop training buttons
		startTraining = new JButton("Start Training");
		startTraining.setFocusPainted(false);
		startTraining.setFocusable(false);
		startTraining.setForeground(new Color(0,171,103));
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
		c.fill = GridBagConstraints.HORIZONTAL;
		c.ipady = 0;
		c.ipadx = 0;
		c.gridwidth = 2;
		c.gridheight = 1;
		c.weightx = 1;
		c.weighty = 0;
		c.gridx = 0;
		c.gridy = 0;
		
		this.add(demoDataPanel,c);
		c.fill = GridBagConstraints.NONE;
		demoDataPanel.add(mnist, c);
		c.fill = GridBagConstraints.HORIZONTAL;
		c.gridy++;
		demoDataPanel.add(dataDirectory, c);
		c.fill = GridBagConstraints.NONE;
		c.gridy++;
		demoDataPanel.add(loadButton, c);
		
		c.gridy = 1;
		c.gridwidth = 2;
		c.fill = GridBagConstraints.HORIZONTAL;
		this.add(visDataPanel,c);
		c.fill = GridBagConstraints.NONE;
		c.gridy = 0;
		c.ipadx = 2;
		c.ipady = 2;
		visDataPanel.add(showAllClasses, c);
		
		c.ipadx = 10;
		c.ipady = 10;
		c.gridy = 2;
		c.gridwidth = 1;
		c.weighty = 1;
		c.anchor = GridBagConstraints.NORTHEAST;
		this.add(startTraining, c);
		c.gridx++;
		c.anchor = GridBagConstraints.NORTHWEST;
		this.add(stopTraining, c);
		
		this.setLocationRelativeTo(null);
	}
	
	private void initListeners() {
		
		stopTraining.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				trainInterrupt.interrupt();
			}
			
		});
		
		loadButton.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				String coreDir = dataDirectory.getValue();
				if (!coreDir.endsWith(File.separator)) {
					coreDir += File.separator;
				}
				
				if (mnist.isSelected()) {
					
					log.info("Loading MNIST digits...");
					System.setProperty("user.home", coreDir);
					
					long startTime = System.currentTimeMillis();
					
					currentDataSet = "MNIST";
					
					modelParams = new ModelParams();
					modelParams.numColsIn = 28;
					modelParams.numRowsIn = 28;
					modelParams.numClasses = 10;
					
					trainingParams = new TrainingParams();
					
					trainData = DemoData.getTrainData(currentDataSet,trainingParams);
					testData = DemoData.getTestData(currentDataSet,trainingParams);
					
					if (trainData==null) {
						log.error("Failed to load MNIST train data.");
					} else if (testData==null) {
						log.error("Failed to load MNIST test data.");
					} else {
						log.info("Loaded MNIST digits in " + Long.toString(System.currentTimeMillis() - startTime) + "ms!");
					}
					
				}
			}
			
		});
		
		startTraining.addActionListener(new ActionListener() {


			@Override
			public void actionPerformed(ActionEvent arg0) {
				thread.run(() -> {trainModel();});
			}			
			
		});
		
		showAllClasses.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				trainData.reset();
				ArrayImg<FloatType,FloatArray> ip = ArrayImgs.floats(modelParams.numColsIn,modelParams.numRowsIn,modelParams.numClasses);
				DataSet example = trainData.next(1);
				for (int i = 0; i<modelParams.numClasses; i++) {
					while (example.outcome()!=i) {
						example = trainData.next(1);
						if (!trainData.hasNext()) {
							trainData.reset();
						}
					}
					if (example.outcome()==i) {
						INDArray img = example.getFeatures();
						IterableInterval<FloatType> ipv = Views.interval(ip,
																		 new long[] {0,0,i},
																		 new long[] {modelParams.numColsIn-1,modelParams.numRowsIn-1,i});
						ConvertersUtility.INDArrayToIIFloat2D(img, ipv);
					}
				}
				ImagePlus imp = ImageJFunctions.show(ip);
				imp.setTitle("Image Classes");
			}
			
		});
	}
	
	private void trainModel() {
        log.info("Build model...");
        ProgressPlotListener listener = new ProgressPlotListener(10, log, 50);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(trainingParams.seed) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006) //specify the learning rate
                .updater(Updater.NESTEROVS).momentum(0.9) //specify the rate of change of the learning rate.
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
                        .nIn(modelParams.numRowsIn * modelParams.numColsIn)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .nIn(1000)
                        .nOut(modelParams.numClasses)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .name("objective")
                        .build())
                .pretrain(false).backprop(true) //use backpropagation to adjust weights
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(listener);

        String tempDir = System.getProperty("user.home");
        //String exampleDirectory = FilenameUtils.concat(tempDir, "TrainedNetworks/");
        EarlyStoppingModelSaver saver = new LocalFileModelSaver(tempDir);
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(50)) //Max of 50 epochs
                .evaluateEveryNEpochs(1)
                .iterationTerminationConditions(trainInterrupt) //Max of 20 minutes
                .scoreCalculator(new DataSetLossCalculator(testData, true))     //Calculate test set score
                .modelSaver(saver)
                .build();
        
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,model,trainData);
        trainer.setListener(listener);
        
        trainer.fit();
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
