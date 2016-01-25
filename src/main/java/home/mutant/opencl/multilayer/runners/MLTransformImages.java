package home.mutant.opencl.multilayer.runners;


import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.Utils;
import home.mutant.opencl.multilayer.TransformImages;
import home.mutant.opencl.smooth.fourd.LinkedClusterablesOpenCl4D;

public class MLTransformImages {

	public static void main(String[] args) throws Exception {
		MnistDatabase.loadImages();
		LinkedClusterablesOpenCl4D filters = (LinkedClusterablesOpenCl4D) Utils.load("smoothclusters4_256_4D");
		long t0=System.currentTimeMillis();
		TransformImages  ti = new TransformImages(MnistDatabase.trainImages.subList(0, 25600), filters,2);
		ti.transform();
		long t=System.currentTimeMillis()-t0;
		
		ResultFrame frame = new ResultFrame(1600, 1000);
		frame.showImages(ti.getTransformedImages().subList(0, 256),16);
		System.out.println(t/1000.+" sec");
	}
}
