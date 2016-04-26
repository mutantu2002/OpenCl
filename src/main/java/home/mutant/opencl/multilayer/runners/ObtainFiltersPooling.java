package home.mutant.opencl.multilayer.runners;

import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.multilayer.steps.ObtainFilters;

public class ObtainFiltersPooling {

 	public static void main(String[] args) throws Exception {
 		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
 		MnistDatabase.loadImages();

		long t0=System.currentTimeMillis();
		int noIterations=50;
		ObtainFilters  of = new ObtainFilters(MnistDatabase.trainImages.subList(0, 60000), 5, 16, noIterations, 1, true);
		of.cluster();
		long t=System.currentTimeMillis()-t0;
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImages(of.getClusterImages(),4);
		System.out.println(1000.*noIterations/t+" it/sec");
 	}
}
