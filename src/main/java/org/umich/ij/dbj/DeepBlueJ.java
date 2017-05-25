package org.umich.ij.dbj;

import javax.swing.SwingUtilities;

import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.command.CommandService;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.thread.ThreadService;
import org.scijava.ui.UIService;

import net.imagej.ops.OpService;

@Plugin(type = Command.class, headless = true,
		menuPath = "Experimental>Deep Blue J")
public class DeepBlueJ implements Command {

	@Parameter
	OpService ops;
	
	@Parameter
	LogService log;
	
	@Parameter
	UIService ui;
	
	@Parameter
	CommandService cmd;	
	
	@Parameter
	StatusService status;
	
	@Parameter
	ThreadService thread;
	
	private static DeepBlueJGUI dialog = null;
	
	@Override
	public void run() {
		// TODO Auto-generated method stub
		SwingUtilities.invokeLater(() -> {
			if (dialog == null) {
				dialog = new DeepBlueJGUI();
			}
			dialog.setVisible(true);
			
			dialog.setOps(ops);
			dialog.setLog(log);
			dialog.setStatus(status);
			dialog.setCommand(cmd);
			dialog.setThread(thread);
			dialog.setUi(ui);
		});
	}
}
