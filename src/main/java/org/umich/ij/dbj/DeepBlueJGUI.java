package org.umich.ij.dbj;

import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.io.File;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JPanel;
import javax.swing.JToggleButton;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;

import org.scijava.app.StatusService;
import org.scijava.command.CommandService;
import org.scijava.log.LogService;
import org.scijava.thread.ThreadService;
import org.scijava.ui.UIService;
import org.umich.ij.guitools.DirectoryChooserPanel;

import net.imagej.ops.OpService;

public class DeepBlueJGUI extends JDialog{
	
	// SciJava/ImageJ parameters
	private OpService ops;
	private LogService log;
	private StatusService status;
	private CommandService cmd;
	private ThreadService thread;
	private UIService ui;
	
	// Setup panels and subpanels
	//private JPanel demoPanel;
	private JPanel downloadSubPanel;
	
	// Download Panel Components
	private DirectoryChooserPanel downloadDirectory;
	private JButton downloadButton;
	private JToggleButton downloadMnist;
	
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
		downloadSubPanel = new JPanel(new GridBagLayout());
			downloadMnist = new JToggleButton("MNIST");
			downloadMnist.setToolTipText("Download the MNIST digits dataset.");
			downloadMnist.setFocusPainted(false);
			downloadDirectory = new DirectoryChooserPanel("Save Directory: ",
														  ij.IJ.getDirectory("home") + "DeepBlueJ Demo Data" + File.separator,
														  20);
			downloadDirectory.setToolTipText("Select a directory to download and unpack data to.");
			downloadButton = new JButton("Download Data");
	}
	
	private void drawElements() {
		// Setup the constraints
		GridBagConstraints c = new GridBagConstraints();
		c.insets = new Insets(1,1,1,1);
		c.anchor = GridBagConstraints.NORTH;
		c.fill = GridBagConstraints.HORIZONTAL;
		c.ipady = 0;
		c.ipadx = 0;
		c.gridwidth = 1;
		c.weightx = 1;
		c.gridx = 0;
		c.gridy = 0;
		
		this.add(downloadSubPanel,c);
		c.fill = GridBagConstraints.NONE;
		downloadSubPanel.add(downloadMnist, c);
		c.fill = GridBagConstraints.HORIZONTAL;
		c.gridy++;
		downloadSubPanel.add(downloadDirectory, c);
		c.fill = GridBagConstraints.NONE;
		c.gridy++;
		downloadSubPanel.add(downloadButton, c);
		
		this.setLocationRelativeTo(null);
	}
	
	private void initListeners() {
		
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
