package home.mutant.opencl.multilayer.runners;


import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;

public class ML {

	public static void main(String[] args) throws Exception {
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImages();

	}
}
