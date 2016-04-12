package home.mutant.opencl.multilayer.runners;


import java.util.List;

import home.mutant.dl.models.Image;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.multilayer.LastLayer;
import home.mutant.opencl.multilayer.OneLayer;

public class ML {

	public static void main(String[] args) throws Exception {
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImagesScaled(0.5);
		OneLayer ol = new OneLayer(MnistDatabase.trainImages,256,1,1,2,2);
		ol.transform();
		List<Image> testImages = ol.transform(MnistDatabase.testImages);
		System.out.println("Last layer");
		ResultFrame framef = new ResultFrame(800, 800);
		framef.showImages(ol.getFilters(), 16);
		
		ResultFrame frame = new ResultFrame(800, 800);
		frame.showImages(ol.getOutImages().subList(0, 256), 16);
		LastLayer ll = new LastLayer(ol.getOutImages(), testImages, MnistDatabase.trainLabels, MnistDatabase.testLabels, 6000, 20);
		ResultFrame frame3 = new ResultFrame(1600, 800);
		frame3.showImages(ll.test());
	}
}
