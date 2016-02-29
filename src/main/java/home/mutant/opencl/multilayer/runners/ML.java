package home.mutant.opencl.multilayer.runners;


import java.util.List;

import home.mutant.dl.models.Image;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.multilayer.LastLayer;
import home.mutant.opencl.multilayer.OneLayer;
import home.mutant.opencl.multilayer.steps.MeanPollingImages;

public class ML {

	public static void main(String[] args) throws Exception {
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImages();
		OneLayer ol = new OneLayer(MnistDatabase.trainImages);
		ol.transform();
		List<Image> testImages = ol.transform(MnistDatabase.testImages);
		System.out.println("Last layer");
		ResultFrame framef = new ResultFrame(800, 800);
		framef.showImages(ol.getFilters(), 16);
		MeanPollingImages mp = new MeanPollingImages(ol.getOutImages());
		mp.transform();
		
		MeanPollingImages mpt = new MeanPollingImages(testImages);
		mpt.transform();
		
		ResultFrame frame = new ResultFrame(800, 800);
		frame.showImages(mp.getTransformedImages().subList(0, 256), 16);
		LastLayer ll = new LastLayer(mp.getTransformedImages(), mpt.getTransformedImages(), MnistDatabase.trainLabels, MnistDatabase.testLabels, 1000, 50);
		ResultFrame frame3 = new ResultFrame(1600, 800);
		frame3.showImages(ll.test());
	}
}
