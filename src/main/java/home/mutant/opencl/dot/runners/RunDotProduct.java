package home.mutant.opencl.dot.runners;

import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.dot.steps.ObtainFiltersDot;
import home.mutant.opencl.dot.steps.TransformImagesMapDot1D;

public class RunDotProduct {

 	public static void main(String[] args) throws Exception {
 		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
 		MnistDatabase.loadImages();

		long t0=System.currentTimeMillis();
		int noIterations=5;
		int noClusters = 16;
		ObtainFiltersDot  of = new ObtainFiltersDot(MnistDatabase.trainImages.subList(0, 256*39)).
				setDimFilterX(5).setDimFilterY(2).setNoClusters(noClusters).build();
		of.cluster();
		long t=System.currentTimeMillis()-t0;
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImages(of.getClusterImages(),(int) Math.sqrt(noClusters));
		System.out.println(1000.*noIterations/t+" it/sec");
		of.constructNormalizedImageClusters();
		TransformImagesMapDot1D ti = new TransformImagesMapDot1D(MnistDatabase.trainImages.subList(0, 256*39), of.getClusterImages())
				.setStrideX(1).setStrideY(1).build();
		ti.transform();
		ResultFrame frame2 = new ResultFrame(1600, 800);
		frame2.showImages(ti.getTransformedImages().subList(0, 16),4);

 	}
}
