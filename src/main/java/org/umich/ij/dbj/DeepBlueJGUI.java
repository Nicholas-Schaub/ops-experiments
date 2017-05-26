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
import javax.swing.JPanel;
import javax.swing.JToggleButton;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
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

import net.imagej.ops.OpService;

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
	
	
	// Start/Stop Network Buttons
	private JButton startTraining;
	private JButton stopTraining;
	
	// DL4J Settings
	private String currentDataSet;
	private DataSetIterator trainData;
	private DataSetIterator testData;
	private NetworkParams netParams;
	
	public DeepBlueJGUI() {
		
		DeepBlueJGUI.setDefaultLookAndFeelDecorated(true);
		
		this.setTitle("DeepBlueJ - Alpha");
		this.setSize(new Dimension(501,283));
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
			// TODO Auto-generated catch block
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
		c.weightx = 1;
		c.weighty = 1;
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
		
		c.ipadx = 10;
		c.ipady = 10;
		c.gridy = 1;
		c.gridwidth = 1;
		c.anchor = GridBagConstraints.NORTHEAST;
		this.add(startTraining, c);
		c.gridx++;
		c.anchor = GridBagConstraints.NORTHWEST;
		this.add(stopTraining, c);
		
		this.setLocationRelativeTo(null);
	}
	
	private void initListeners() {
		
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
					
					netParams = new NetworkParams();
					netParams.numColsIn = 28;
					netParams.numRowsIn = 28;
					netParams.numClasses = 10;
					
					trainData = DemoData.getTrainData(currentDataSet,netParams);
					testData = DemoData.getTestData(currentDataSet,netParams);
					
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
			public void actionPerformed(ActionEvent e) {
		        log.info("Build model...");
		        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		                .seed(netParams.seed) //include a random seed for reproducibility
		                // use stochastic gradient descent as an optimization algorithm
		                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		                .iterations(1)
		                .learningRate(0.006) //specify the learning rate
		                .updater(Updater.NESTEROVS).momentum(0.9) //specify the rate of change of the learning rate.
		                .regularization(true).l2(1e-4)
		                .list()
		                .layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
		                        .nIn(netParams.numRowsIn * netParams.numColsIn)
		                        .nOut(1000)
		                        .activation(Activation.RELU)
		                        .weightInit(WeightInit.XAVIER)
		                        .build())
		                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
		                        .nIn(1000)
		                        .nOut(netParams.numClasses)
		                        .activation(Activation.SOFTMAX)
		                        .weightInit(WeightInit.XAVIER)
		                        .build())
		                .pretrain(false).backprop(true) //use backpropagation to adjust weights
		                .build();
		        MultiLayerNetwork model = new MultiLayerNetwork(conf);
		        model.init();
		        model.setListeners(new ScoreIterationListener(100));
		        
		        log.info("Training model...");
		        for( int i=0; i<netParams.numEpochs; i++ ){
		            model.fit(trainData);
		        }
		        
		        log.info("Evaluate model...");
		        Evaluation eval = new Evaluation(netParams.numClasses); //create an evaluation object with 10 possible classes
		        while(testData.hasNext()){
		            DataSet next = testData.next();
		            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
		            eval.eval(next.getLabels(), output); //check the prediction against the true class
		        }

		        log.info(eval.stats());
			}
			
		});
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
