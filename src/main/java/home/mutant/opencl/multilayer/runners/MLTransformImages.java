package home.mutant.opencl.multilayer.runners;


import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.opencl.multilayer.ArrangeFilters;
import home.mutant.opencl.multilayer.TransformImages;

public class MLTransformImages {

	public static void main(String[] args) throws Exception {
		MnistDatabase.loadImages();
		long t0=System.currentTimeMillis();
		TransformImages  ti = new TransformImages(MnistDatabase.trainImages, new ArrangeFilters(null),1);
		ti.transform();
		long t=System.currentTimeMillis()-t0;
		
		ResultFrame frame = new ResultFrame(1600, 1000);
		frame.showImages(ti.getTransformedImages().subList(10000, 10256),16);
		System.out.println(t/1000.+" sec");
	}
}
