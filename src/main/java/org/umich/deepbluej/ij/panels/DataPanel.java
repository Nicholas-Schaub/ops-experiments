package org.umich.deepbluej.ij.panels;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JToggleButton;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.umich.deepbluej.TrainingParams;
import org.umich.deepbluej.ij.plugin.DemoData;
import org.umich.ij.guitools.DirectoryChooserPanel;

import ij.ImagePlus;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imglib2.IterableInterval;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

public class DataPanel extends JPanel {
	// Demo data set load panel & components
	private JPanel demoDataPanel;
	private JPanel demoDatasetsPanel;
	private JToggleButton mnist;
	private JPanel demoDataInfoPanel;
	private JLabel demoDataNameLabel;
	private JLabel demoDataName;
	private JLabel demoNumTrainLabel;
	private JLabel demoNumTrain;
	private JLabel demoNumTestLabel;
	private JLabel demoNumTest;
	private DirectoryChooserPanel dataDirectory;
	private JButton loadButton;
	private JButton showAllClasses;
	
	private String currentDataSet;
	private TrainingParams trainingParams;
	
	public DataPanel(TrainingParams trainingParams) {
		this.trainingParams = trainingParams;
		initElements();
		drawElements();
		initListeners();
	}
	
	private void initElements() {
		// Create the download subpanel
		demoDataPanel = new JPanel(new GridBagLayout());
		demoDataPanel.setBorder(BorderFactory.createTitledBorder("Demo Data"));
			demoDatasetsPanel = new JPanel(new GridBagLayout());
			demoDatasetsPanel.setBorder(BorderFactory.createTitledBorder("Available Datasets"));
				mnist = new JToggleButton("MNIST");
				mnist.setToolTipText("Download and load the MNIST digits dataset.");
				mnist.setFocusPainted(false);
				mnist.setFocusable(false);
				dataDirectory = new DirectoryChooserPanel("Save Directory: ",
						  ij.IJ.getDirectory("home") + "DeepBlueJ Demo Data" + File.separator,
						  25);
				dataDirectory.setToolTipText("Select a directory to download and unpack data to.");
				loadButton = new JButton("Load Data");
				loadButton.setFocusPainted(false);
				loadButton.setFocusable(false);
			demoDataInfoPanel = new JPanel(new GridBagLayout());
			demoDataInfoPanel.setBorder(BorderFactory.createTitledBorder("Current Dataset Information"));
				demoDataNameLabel = new JLabel("Dataset Name: ");
				demoDataName = new JLabel("None");
				demoNumTrainLabel = new JLabel("# Train Images: ");
				demoNumTrain = new JLabel("None");
				demoNumTestLabel = new JLabel("# Test Images: ");
				demoNumTest = new JLabel("None");
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
		// Demo data panel
		//
		this.add(demoDataPanel,c);
		c.fill = GridBagConstraints.BOTH;
		demoDataPanel.add(demoDatasetsPanel,c);
		c.gridy++;
		demoDataPanel.add(demoDataInfoPanel, c);
		c.gridy = 0;
		c.fill = GridBagConstraints.NONE;
		demoDatasetsPanel.add(mnist, c);
		c.fill = GridBagConstraints.HORIZONTAL;
		c.gridy++;
		demoDatasetsPanel.add(dataDirectory, c);
		c.fill = GridBagConstraints.NONE;
		c.gridy++;
		demoDatasetsPanel.add(loadButton, c);
		c.gridy = 0;
		c.gridx = 0;
		c.gridwidth = 1;
		c.anchor = GridBagConstraints.EAST;
		demoDataInfoPanel.add(demoDataNameLabel, c);
		c.gridx++;
		c.anchor = GridBagConstraints.WEST;
		demoDataInfoPanel.add(demoDataName, c);
		c.gridx++;
		c.anchor = GridBagConstraints.EAST;
		demoDataInfoPanel.add(demoNumTrainLabel, c);
		c.gridx++;
		c.anchor = GridBagConstraints.WEST;
		demoDataInfoPanel.add(demoNumTrain, c);
		c.gridx++;
		c.anchor = GridBagConstraints.EAST;
		demoDataInfoPanel.add(demoNumTestLabel, c);
		c.gridx++;
		c.anchor = GridBagConstraints.WEST;
		demoDataInfoPanel.add(demoNumTest, c);
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
					
					System.setProperty("user.home", coreDir);
					
					long startTime = System.currentTimeMillis();

					currentDataSet = "MNIST";
					
					trainingParams.trainData(DemoData.getTrainData(currentDataSet,trainingParams));
					trainingParams.testData(DemoData.getTestData(currentDataSet,trainingParams));
					
					trainingParams.dataName = currentDataSet;
					
					if (trainingParams.trainData()==null) {
						System.out.println("Failed to load MNIST train data.");
					} else if (trainingParams.testData()==null) {
						System.out.println("Failed to load MNIST test data.");
					} else {
						System.out.println("Loaded MNIST digits in " + Long.toString(System.currentTimeMillis() - startTime) + "ms!");
					}
				}
			}
			
		});
		
		showAllClasses.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				trainingParams.trainData().reset();
				ArrayImg<FloatType,FloatArray> ip = ArrayImgs.floats(trainingParams.colsIn(),trainingParams.rowsIn(),trainingParams.attributesOut());
				DataSet example = trainingParams.trainData().next(1);
				for (int i = 0; i<trainingParams.attributesOut(); i++) {
					while (example.outcome()!=i) {
						example = trainingParams.trainData().next(1);
						if (!trainingParams.trainData().hasNext()) {
							trainingParams.trainData().reset();
						}
					}
					if (example.outcome()==i) {
						INDArray img = example.getFeatures();
						IterableInterval<FloatType> ipv = Views.interval(ip,
																		 new long[] {0,0,i},
																		 new long[] {trainingParams.colsIn()-1,trainingParams.rowsIn()-1,i});
						ConvertersUtility.INDArrayToIIFloat2D(img, ipv);
					}
				}
				ImagePlus imp = ImageJFunctions.show(ip);
				imp.setTitle("Image Classes");
			}
			
		});
	}
}
