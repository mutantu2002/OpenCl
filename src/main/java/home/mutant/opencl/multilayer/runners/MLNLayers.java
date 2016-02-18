package home.mutant.opencl.multilayer.runners;


import java.util.List;

import home.mutant.dl.models.Image;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.multilayer.LastLayer;
import home.mutant.opencl.multilayer.OneLayer;

public class MLNLayers {

	public static void main(String[] args) throws Exception {
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImagesPadded(6);
		OneLayer ol = new OneLayer(MnistDatabase.trainImages,1024,2,1,3,3);
		ol.transform();
		List<Image> testImages = ol.transform(MnistDatabase.testImages);
		
		ResultFrame frame = new ResultFrame(800, 800);
		frame.showImages(ol.getOutImages().subList(0, 256), 16);
		
		for(int l=0;l<6;l++){
			OneLayer ol2 = new OneLayer(ol.getOutImages(),1024,2,2,4,2);
			ol2.transform();
			testImages = ol2.transform(testImages);
			ol=ol2;
		}
		
		System.out.println("Last layer");
		ResultFrame frame2 = new ResultFrame(800, 800);
		frame2.showImages(ol.getOutImages().subList(0, 256), 16);
		LastLayer ll = new LastLayer(ol.getOutImages(), testImages, MnistDatabase.trainLabels, MnistDatabase.testLabels, 2000, 50);
		ll.test();
	}
}
