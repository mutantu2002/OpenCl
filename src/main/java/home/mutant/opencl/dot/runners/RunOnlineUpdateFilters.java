package home.mutant.opencl.dot.runners;

import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.dot.steps.OnlineUpdateFilters;

public class RunOnlineUpdateFilters {

 	public static void main(String[] args) throws Exception {
 		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
 		MnistDatabase.loadImages();

		long t0=System.currentTimeMillis();
		int noIterations=5;
		int noClusters = 256;
		OnlineUpdateFilters  of = new OnlineUpdateFilters(MnistDatabase.trainImages.subList(0, 256*39)).
				setDimFilter(7).setNoClusters(noClusters).build();
		of.cluster();
		long t=System.currentTimeMillis()-t0;
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImages(of.getClusterImages(),(int) Math.sqrt(noClusters));
		System.out.println(1000.*noIterations/t+" it/sec");
 	}
}
