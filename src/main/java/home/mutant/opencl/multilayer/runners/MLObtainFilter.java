package home.mutant.opencl.multilayer.runners;


import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.multilayer.ObtainFilters;

public class MLObtainFilter {

	public static void main(String[] args) throws Exception {
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImages();
		long t0=System.currentTimeMillis();
		int noIterations=10;
		ObtainFilters  ci = new ObtainFilters(MnistDatabase.trainImages, 4, 256, noIterations);
		ci.cluster();
		long t=System.currentTimeMillis()-t0;
		
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImages(ci.getClusterImages(),16);
		System.out.println(1000.*noIterations/t+" it/sec");
	}
}
