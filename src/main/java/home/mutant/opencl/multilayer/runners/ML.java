package home.mutant.opencl.multilayer.runners;


import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.multilayer.ClusterImages;

public class ML {

	public static void main(String[] args) throws Exception {
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImages();
		long t0=System.currentTimeMillis();
		int noIterations=100;
		ClusterImages  ci = new ClusterImages(MnistDatabase.trainImages, MnistDatabase.trainLabels, 2000, noIterations);
		ci.cluster();
		long t=System.currentTimeMillis()-t0;
		ci.test(MnistDatabase.testImages, MnistDatabase.testLabels);
		ci.releaseOpenCl();
		
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImages(ci.getClusters());
		System.out.println(1000.*noIterations/t+" it/sec");
	}
}
