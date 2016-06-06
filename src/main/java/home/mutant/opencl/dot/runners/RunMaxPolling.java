package home.mutant.opencl.dot.runners;

import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.dot.steps.MaxPolling;

public class RunMaxPolling {

 	public static void main(String[] args) throws Exception {
 		MnistDatabase.IMAGE_TYPE = TYPE.SHORT;
 		MnistDatabase.loadImages();

		long t0=System.currentTimeMillis();
		MaxPolling  mp = new MaxPolling(MnistDatabase.trainImages.subList(0, 256*3));
		mp.transform();
		long t=System.currentTimeMillis()-t0;
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImages(mp.getTransformedImages().subList(0, 16),4);
		System.out.println(t/1000.+" sec");
 	}
}
