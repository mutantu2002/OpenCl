package home.mutant.opencl.multilayer.runners;


import java.util.List;

import home.mutant.dl.models.Image;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.multilayer.LastLayer;
import home.mutant.opencl.multilayer.OneLayer;

public class ML2Layers {

	public static void main(String[] args) throws Exception {
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImages();
		OneLayer ol = new OneLayer(MnistDatabase.trainImages);
		ol.transform();
		List<Image> testImages = ol.transform(MnistDatabase.testImages);
		
		ResultFrame frame = new ResultFrame(800, 800);
		frame.showImages(ol.getOutImages().subList(0, 256), 16);
		
//		MeanPollingImages mp = new MeanPollingImages(ol.getOutImages());
//		mp.transform();
//		
//		MeanPollingImages mpt = new MeanPollingImages(testImages);
//		mpt.transform();
		
		OneLayer ol2 = new OneLayer(ol.getOutImages(),1024,2,2,4,2);
		ol2.transform();
		testImages = ol2.transform(testImages);
		
		ResultFrame framef = new ResultFrame(800, 800);
		framef.showImages(ol2.getFilters().subList(0, 256), 16);
		
		System.out.println("Last layer");
		ResultFrame frame2 = new ResultFrame(800, 800);
		frame2.showImages(ol2.getOutImages().subList(0, 256), 16);
		LastLayer ll = new LastLayer(ol2.getOutImages(), testImages, MnistDatabase.trainLabels, MnistDatabase.testLabels, 4000, 50);
		ll.test();
	}
}
