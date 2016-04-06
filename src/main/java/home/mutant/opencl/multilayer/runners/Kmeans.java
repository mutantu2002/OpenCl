package home.mutant.opencl.multilayer.runners;

import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.multilayer.steps.ClusterImages;

public class Kmeans {

 	public static void main(String[] args) throws Exception {
 		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
 		MnistDatabase.loadImagesCrop(20);;

		long t0=System.currentTimeMillis();
		int noIterations=20;
		ClusterImages  ci = new ClusterImages(MnistDatabase.trainImages, MnistDatabase.trainLabels, 10000, noIterations);
		ci.cluster();
		ci.test(MnistDatabase.testImages, MnistDatabase.testLabels);
		ci.releaseOpenCl();
		long t=System.currentTimeMillis()-t0;
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImages(ci.getClusters().subList(0, 256),16);
		System.out.println(1000.*noIterations/t+" it/sec");
 	}
}
