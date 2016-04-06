package home.mutant.opencl.multilayer.map.runners;


import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.multilayer.steps.ObtainFilters;
import home.mutant.opencl.multilayer.steps.TransformImagesMap;

public class MLMap {

	public static void main(String[] args) throws Exception {
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImages();
		ObtainFilters of = new ObtainFilters(MnistDatabase.trainImages, 5, 64, 30);
		of.cluster();
		ResultFrame frame = new ResultFrame(800, 800);
		frame.showImages(of.getClusterImages(),8);
		
		TransformImagesMap tm = new TransformImagesMap(MnistDatabase.trainImages, of.getClusterImages(), 1);
		tm.transform();
		ResultFrame frame2 = new ResultFrame(1200, 900);
		frame2.showImages(tm.getTransformedImages().subList(0, 36), 6);
	}

}
