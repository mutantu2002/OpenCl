package home.mutant.opencl.dot.runners;

import java.util.ArrayList;
import java.util.List;

import home.mutant.dl.models.Image;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.dot.steps.MaxIndexImage;
import home.mutant.opencl.dot.steps.ObtainFiltersDotShort;
import home.mutant.opencl.dot.steps.TransformImagesMapDot1DShort;

public class RunDotProductShort {

 	public static void main(String[] args) throws Exception {
 		MnistDatabase.IMAGE_TYPE = TYPE.SHORT;
 		MnistDatabase.loadImages();

		List<List<Image>> imagesByType= MnistDatabase.getImagesByType();
		
		int noIterations=40;
		int noClusters = 150;
		List<Image> clusters = new ArrayList<>();
		for (int i=0;i<10;i++){
			ObtainFiltersDotShort  of = new ObtainFiltersDotShort(imagesByType.get(i)).
					setDimFilterX(25).setDimFilterY(25).setNoClusters(noClusters)
					.setNoIterations(noIterations).setRandomizeFromData(true).build();
			of.cluster();
			of.constructNormalizedImageClusters();
			clusters.addAll(of.getClusterImages());
		}
		TransformImagesMapDot1DShort ti = new TransformImagesMapDot1DShort(MnistDatabase.testImages.subList(0, 256*39),clusters )
				.setStrideX(1).setStrideY(1).build();
		ti.transform();
		ResultFrame frame2 = new ResultFrame(1600, 800);
		frame2.showImages(ti.getTransformedImages().subList(0, 16),16);
		MaxIndexImage mii = new MaxIndexImage(ti.getTransformedImages(),MnistDatabase.testLabels.subList(0, 256*39),noClusters*16);
		System.out.println(mii.rate);
		
		
 	}
}
