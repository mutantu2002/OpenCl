package home.mutant.opencl.dot.runners;

import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.dot.steps.GradientClustering;

public class RunGradientClustering {

 	public static void main(String[] args) throws Exception {
 		MnistDatabase.IMAGE_TYPE = TYPE.SHORT;
 		MnistDatabase.loadImages();

		long t0=System.currentTimeMillis();
		GradientClustering gc = new GradientClustering(MnistDatabase.trainImages.subList(0, 256*234),1) .build();
		gc.cluster();
		long t=System.currentTimeMillis()-t0;
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImages(gc.getPerceptronImages(),10);
		System.out.println(t/1000.+" sec");
 	}
}
