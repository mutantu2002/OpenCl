package home.mutant.opencl.dot.runners;

import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.dot.steps.SoftMaxShort;

public class RunSoftMax {

 	public static void main(String[] args) throws Exception {
 		MnistDatabase.IMAGE_TYPE = TYPE.SHORT;
 		MnistDatabase.loadImages();

		long t0=System.currentTimeMillis();
		SoftMaxShort sm = new SoftMaxShort(MnistDatabase.trainImages.subList(0, 256*234), 
				MnistDatabase.trainLabels.subList(0, 256*234), 10).setNoIterations(100).build();
		sm.cluster();
		long t=System.currentTimeMillis()-t0;
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImages(sm.getPerceptronImages(),10);
		System.out.println(t/1000.+" sec");
		
		double rate = sm.test(MnistDatabase.testImages.subList(0, 256*39), MnistDatabase.testLabels.subList(0, 256*39));
		System.out.println(rate);
 	}
}
