package home.mutant.opencl.multilayer.map.runners;


import java.util.List;

import home.mutant.dl.models.Image;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.multilayer.steps.ObtainFilters;
import home.mutant.opencl.multilayer.steps.TransformImagesMap;

public class MLMap {
	public static List<Image> images;
	public static void main(String[] args) throws Exception {
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImages();
		images=MnistDatabase.trainImages.subList(0, 256*3);
//		step(5,36,1);
//		step(12,25,6);
//		step(10,16,5);
//		step(8,9,4);
//		step(6,4,3);
		step(4,25,1);
		for(int i=0;i<6;i++)
			step(15,25,5);
	}
	
	public static void step(int dimFilter, int noClusters, int stride){
		ObtainFilters of = new ObtainFilters(images, dimFilter, noClusters, 20, stride, false);
		of.cluster();
		ResultFrame frame = new ResultFrame(800, 800);
		frame.showImages(of.getClusterImages(),(int) Math.sqrt(of.getClusterImages().size()));
		
		TransformImagesMap tm = new TransformImagesMap(images, of.getClusterImages(), stride);
		tm.transform();
		ResultFrame frame2 = new ResultFrame(1200, 900);
		frame2.showImages(tm.getTransformedImages().subList(0, 16), 4);
		images = tm.getTransformedImages();
	}
}
