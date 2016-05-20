package home.mutant.opencl.dot.runners;

import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.dot.steps.ObtainFiltersDot;

public class ObtainFiltersDotProduct {

 	public static void main(String[] args) throws Exception {
 		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
 		MnistDatabase.loadImages();

		long t0=System.currentTimeMillis();
		int noIterations=2;
		int noClusters = 36;
		ObtainFiltersDot  of = new ObtainFiltersDot(MnistDatabase.trainImages.subList(0, 256*234), 14, noClusters, noIterations, 1);
		of.cluster();
		long t=System.currentTimeMillis()-t0;
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImages(of.getClusterImages(),(int) Math.sqrt(noClusters));
		System.out.println(1000.*noIterations/t+" it/sec");
 	}
}
