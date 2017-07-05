package org.umich.deepbluej.ij.panels;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import org.umich.deepbluej.ModelParams;
import org.umich.ij.guitools.ValidatedTextField;
import org.umich.ij.guitools.ValidatorInt;

public class ModelPanel extends JPanel {
	ModelParams modelParams;
	
	// Model panel
	private JPanel modelPanel;
	private JLabel modelTypeLabel;
	private JComboBox<String> modelType;
	private JLabel modelDepthLabel;
	private ValidatedTextField<Integer> modelDepth;
	private JLabel modelDepthMaxLabel;
	private JLabel modelDepthMax;
	private JLabel modelNumAttributesLabel;
	private ValidatedTextField<Integer> modelNumAttributes;
	private JLabel modelInpWidthLabel;
	private ValidatedTextField<Integer> modelInpWidth;
	private JLabel modelInpHeightLabel;
	private ValidatedTextField<Integer> modelInpHeight;
	private JLabel modelInpAttributesLabel;
	private ValidatedTextField<Integer> modelInpAttributes;
	private JLabel modelOutWidthLabel;
	private ValidatedTextField<Integer> modelOutWidth;
	private JLabel modelOutHeightLabel;
	private ValidatedTextField<Integer> modelOutHeight;
	private JLabel modelUnitTypeLabel;
	private JComboBox<String> modelUnitType;
	private JLabel modelUnitScaleLabel;
	private JTextField modelUnitScale;
	private JLabel modelUnitComplexityLabel;
	private ValidatedTextField<Integer> modelUnitComplexity;
	private JLabel modelUnitRepeatLabel;
	private JTextField modelUnitRepeat;
	private JButton updateSettings;
	
	public ModelPanel(ModelParams modelParams) {
		this.modelParams = modelParams;
		initElements();
		drawElements();
		initListeners();
	}
	
	private void initElements() {

		modelPanel = new JPanel(new GridBagLayout());
		modelPanel.setBorder(BorderFactory.createTitledBorder("Model Settings"));
			modelTypeLabel = new JLabel("Model Type: ");
			modelType = new JComboBox<String>(ModelParams.MODEL_TYPE_STRING);
			modelType.setFocusable(false);
			modelDepthLabel = new JLabel("# Scales: ");
			modelDepth = new ValidatedTextField<Integer>(9, Integer.toString(modelParams.modelDepth()),new ValidatorInt(0,modelParams.modelScalesMax()));
			modelDepthMaxLabel = new JLabel("Max Scales: ");
			modelDepthMax = new JLabel(Integer.toString(modelParams.modelScalesMax()));
			modelInpWidthLabel = new JLabel("Width In: ");
			modelInpWidth = new ValidatedTextField<Integer>(9, Integer.toString(modelParams.colsIn()),new ValidatorInt(16,100000));
			modelInpHeightLabel = new JLabel("Height In: ");
			modelInpHeight = new ValidatedTextField<Integer>(9, Integer.toString(modelParams.rowsIn()),new ValidatorInt(16,100000));
			modelInpAttributesLabel = new JLabel("Input Attributes: ");
			modelInpAttributes = new ValidatedTextField<Integer>(9, Integer.toString(modelParams.attributesIn()),new ValidatorInt(1,100000));
			modelOutWidthLabel = new JLabel("Width Out: ");
			modelOutWidth = new ValidatedTextField<Integer>(9, Integer.toString(modelParams.colsOut()),new ValidatorInt(1,100000));
			modelOutHeightLabel = new JLabel("Height Out: ");
			modelOutHeight = new ValidatedTextField<Integer>(9, Integer.toString(modelParams.rowsOut()),new ValidatorInt(1,100000));
			modelNumAttributesLabel = new JLabel("# Attributes: ");
			modelNumAttributes = new ValidatedTextField<Integer>(9, Integer.toString(modelParams.attributesOut()),new ValidatorInt(0,100000));
			modelUnitTypeLabel = new JLabel("Unit Type: ");
			modelUnitType = new JComboBox<String>(ModelParams.UNIT_TYPE_STRING);
			modelUnitType.setFocusable(false);
			modelUnitScaleLabel = new JLabel("Unit Scales: ");
			modelUnitScale = new JTextField(modelParams.unitScaleString());
			modelUnitComplexityLabel = new JLabel("Unit Complexity: ");
			modelUnitComplexity = new ValidatedTextField<Integer>(36, Integer.toString(modelParams.unitComplexity()),new ValidatorInt(3,16));
			modelUnitRepeatLabel = new JLabel("Unit Repeat: ");
			modelUnitRepeat = new JTextField(modelParams.unitRepeatString());
			updateSettings = new JButton("Update Settings");
			updateSettings.setFocusPainted(false);
			updateSettings.setFocusable(false);
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
		// Model Panel
		//
		c.gridx = 0;
		c.gridy = 1;
		c.gridwidth = 2;
		c.fill = GridBagConstraints.BOTH;
		c.anchor = GridBagConstraints.NORTH;
		this.add(modelPanel,c);
		
		c.fill = GridBagConstraints.NONE;
		c.gridy = 0;
		c.gridwidth = 1;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.EAST;
		modelPanel.add(modelTypeLabel, c);
		c.anchor = GridBagConstraints.WEST;
		c.gridx++;
		c.ipadx = 0;
		modelPanel.add(modelType, c);
		c.gridx++;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.EAST;
		modelPanel.add(modelDepthLabel, c);
		c.gridx++;
		c.ipadx = 35;
		c.anchor = GridBagConstraints.WEST;
		modelPanel.add(modelDepth, c);
		c.gridx++;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.EAST;
		modelPanel.add(modelDepthMaxLabel, c);
		c.gridx++;
		c.ipadx = 35;
		c.anchor = GridBagConstraints.WEST;
		modelPanel.add(modelDepthMax, c);
		
		c.gridy++;
		c.gridx = 0;
		c.gridwidth = 1;
		c.fill = GridBagConstraints.NONE;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.EAST;
		modelPanel.add(modelInpWidthLabel, c);
		c.gridx++;
		c.ipadx = 35;
		c.anchor = GridBagConstraints.WEST;
		modelPanel.add(modelInpWidth,c);
		c.gridx++;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.EAST;
		modelPanel.add(modelInpHeightLabel, c);
		c.gridx++;
		c.ipadx = 35;
		c.anchor = GridBagConstraints.WEST;
		modelPanel.add(modelInpHeight, c);
		c.gridx++;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.EAST;
		modelPanel.add(modelInpAttributesLabel, c);
		c.gridx++;
		c.ipadx = 35;
		c.anchor = GridBagConstraints.WEST;
		modelPanel.add(modelInpAttributes, c);
		
		c.gridy++;
		c.gridx = 0;
		c.gridwidth = 1;
		c.fill = GridBagConstraints.NONE;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.EAST;
		modelPanel.add(modelOutWidthLabel, c);
		c.gridx++;
		c.ipadx = 35;
		c.anchor = GridBagConstraints.WEST;
		modelPanel.add(modelOutWidth,c);
		c.gridx++;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.EAST;
		modelPanel.add(modelOutHeightLabel, c);
		c.gridx++;
		c.ipadx = 35;
		c.anchor = GridBagConstraints.WEST;
		modelPanel.add(modelOutHeight, c);
		c.gridx++;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.EAST;
		modelPanel.add(modelNumAttributesLabel, c);
		modelNumAttributesLabel.setVisible(false);
		c.gridx++;
		c.ipadx = 35;
		c.anchor = GridBagConstraints.WEST;
		modelPanel.add(modelNumAttributes, c);
		
		c.gridy++;
		c.gridx = 0;
		c.gridwidth = 1;
		c.fill = GridBagConstraints.NONE;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.EAST;
		modelPanel.add(modelUnitTypeLabel, c);
		c.gridx++;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.WEST;
		modelPanel.add(modelUnitType,c);
		c.gridx++;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.EAST;
		modelPanel.add(modelUnitScaleLabel, c);
		c.gridx++;
		c.ipadx = 185;
		c.gridwidth = 3;
		c.anchor = GridBagConstraints.WEST;
		modelPanel.add(modelUnitScale, c);
		
		c.gridy++;
		c.gridx = 0;
		c.gridwidth = 1;
		c.fill = GridBagConstraints.NONE;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.EAST;
		modelPanel.add(modelUnitComplexityLabel, c);
		c.gridx++;
		c.ipadx = 35;
		c.anchor = GridBagConstraints.WEST;
		modelPanel.add(modelUnitComplexity,c);
		c.gridx++;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.EAST;
		modelPanel.add(modelUnitRepeatLabel, c);
		c.gridx++;
		c.ipadx = 185;
		c.gridwidth = 3;
		c.anchor = GridBagConstraints.WEST;
		modelPanel.add(modelUnitRepeat, c);
		
		c.gridy++;
		c.gridx = 2;
		c.gridwidth = 2;
		c.fill = GridBagConstraints.NONE;
		c.ipadx = 0;
		c.anchor = GridBagConstraints.CENTER;
		modelPanel.add(updateSettings, c);
	}
	
	private void initListeners() {
		modelType.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent arg0) {
				updateModelParams();
			}
			
		});
		
		updateSettings.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent arg0) {
				updateModelParams();
			}
			
		});
	}
	
	public void updateModelParams() {
		// Set model parameters that are set in the GUI
		modelParams.modelType(modelType.getSelectedIndex());
		modelParams.colsIn(Integer.parseInt(modelInpWidth.getText()));
		modelParams.rowsIn(Integer.parseInt(modelInpHeight.getText()));
		modelParams.attributesIn(Integer.parseInt(modelInpAttributes.getText()));
		modelParams.attributesOut(Integer.parseInt(modelNumAttributes.getText()));
		modelParams.modelDepth(Integer.parseInt(modelDepth.getText()));
		modelParams.rowsOut(Integer.parseInt(modelOutWidth.getText()));
		modelParams.colsOut(Integer.parseInt(modelOutHeight.getText()));
		modelParams.unitType(modelUnitType.getSelectedIndex());
		modelParams.unitScale(modelUnitScale.getText());
		modelParams.unitRepeat(modelUnitRepeat.getText());
		modelParams.unitComplexity(Integer.parseInt(modelUnitComplexity.getText()));
		
		// Update GUI based on model restrictions
		modelDepth.setText(Integer.toString(modelParams.modelDepth()));
		modelDepthMax.setText(Integer.toString(modelParams.modelScalesMax()));
		modelOutWidth.setText(Integer.toString(modelParams.colsOut()));
		modelOutHeight.setText(Integer.toString(modelParams.rowsOut()));
		modelNumAttributes.setText(Integer.toString(modelParams.attributesOut()));
		modelUnitRepeat.setText(modelParams.unitRepeatString());
		modelUnitScale.setText(modelParams.unitScaleString());
	}
}
