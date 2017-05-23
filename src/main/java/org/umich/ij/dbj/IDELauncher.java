package org.umich.ij.dbj;

import net.imagej.ImageJ;

public class IDELauncher {
	
	public static void main(final String[] args) {
		final ImageJ ij = net.imagej.Main.launch(args);
		
		ij.command().run(DeepBlueJ.class, true);
	}

}
