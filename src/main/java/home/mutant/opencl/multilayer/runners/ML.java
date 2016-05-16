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
		MnistDatabase.loadImages();
		OneLayer ol = new OneLayer(MnistDatabase.trainImages.subList(0, 256*39),256,2,1,5,5, false);
		ol.transform();
		List<Image> testImages = ol.transform(MnistDatabase.testImages.subList(0, 256*39));
		System.out.println("Last layer");
		ResultFrame framef = new ResultFrame(800, 800);
		framef.showImages(ol.getFilters(), 16);
		
		ResultFrame frame = new ResultFrame(800, 800);
		frame.showImages(ol.getOutImages().subList(0, 256), 16);
		LastLayer ll = new LastLayer(ol.getOutImages(), testImages, MnistDatabase.trainLabels, MnistDatabase.testLabels, 500, 50);
		ResultFrame frame3 = new ResultFrame(1600, 800);
		frame3.showImages(ll.test());
	}
}
