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
		int noIterations=10;
		ClusterImages  ci = new ClusterImages(MnistDatabase.trainImages, MnistDatabase.trainLabels, 1000, noIterations);
		ci.cluster();
		ci.releaseOpenCl();
		long t=System.currentTimeMillis()-t0;
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImages(ci.getClusters());
		System.out.println(1000.*noIterations/t+" it/sec");
	}
}
