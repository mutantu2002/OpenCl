package home.mutant.opencl.dot.runners;

import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.dot.steps.ObtainFiltersDotShort;

public class RunDotProductShort {

 	public static void main(String[] args) throws Exception {
 		MnistDatabase.IMAGE_TYPE = TYPE.SHORT;
 		MnistDatabase.loadImages();

		long t0=System.currentTimeMillis();
		int noIterations=100;
		int noClusters = 9;
		ObtainFiltersDotShort  of = new ObtainFiltersDotShort(MnistDatabase.trainImages.subList(0, 256*39)).
				setDimFilterX(4).setDimFilterY(4).setNoClusters(noClusters).setNoIterations(noIterations).build();
		of.cluster();
		long t=System.currentTimeMillis()-t0;
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImages(of.getClusterImages(),(int) Math.sqrt(noClusters));
		System.out.println(1000.*noIterations/t+" it/sec");
 	}
}
