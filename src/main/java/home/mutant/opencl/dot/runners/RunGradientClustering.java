package home.mutant.opencl.dot.runners;

import java.util.List;

import home.mutant.dl.models.Image;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.dot.steps.GradientClustering;

public class RunGradientClustering {

 	public static void main(String[] args) throws Exception {
 		MnistDatabase.IMAGE_TYPE = TYPE.SHORT;
 		MnistDatabase.loadImages();
 		List<List<Image>> imagesByType= MnistDatabase.getImagesByType();
		long t0=System.currentTimeMillis();
		GradientClustering gc = new GradientClustering(imagesByType.get(0),1).setNoIterations(6000).build();
		gc.cluster();
		long t=System.currentTimeMillis()-t0;
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImages(gc.getPerceptronImages(),10);
		System.out.println(t/1000.+" sec");
 	}
}
