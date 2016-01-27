package home.mutant.opencl.multilayer.runners;


import java.util.List;

import home.mutant.dl.models.Image;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.multilayer.LastLayer;
import home.mutant.opencl.multilayer.OneLayer;

public class ML {

	public static void main(String[] args) throws Exception {
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImages();
		OneLayer ol = new OneLayer(MnistDatabase.trainImages);
		ol.transform();
		List<Image> testImages = ol.transform(MnistDatabase.testImages);
		System.out.println("Last layer");
		LastLayer ll = new LastLayer(ol.getOutImages(), testImages, MnistDatabase.trainLabels, MnistDatabase.testLabels, 1000, 50);
		ll.test();
	}
}
