package home.mutant.opencl.multilayer.runners;


import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.multilayer.ArrangeFilters;
import home.mutant.opencl.multilayer.ClusterImages;
import home.mutant.opencl.multilayer.ObtainFilters;
import home.mutant.opencl.multilayer.TransformImages;

public class MLObtainFilter {
	public static final int FRAMES = 1000;
	public static void main(String[] args) throws Exception {
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImages();
		long t0=System.currentTimeMillis();
		int noIterations=30;
		ObtainFilters  of = new ObtainFilters(MnistDatabase.trainImages, 4, 256, noIterations);
		of.cluster();
		long t=System.currentTimeMillis()-t0;
		System.out.println(1000.*noIterations/t+" it/sec");
		ResultFrame frame = new ResultFrame(300, 300);
		frame.showImages(of.getClusterImages(),16);
		
		ArrangeFilters af = new ArrangeFilters(of.getClusterImages());
		af.listDistances();
		t0 = System.currentTimeMillis();
		for(int i=0;i<FRAMES;i++){
			af.stepV();
			af.show();
			if(i%100==0)System.out.println(i);
		}
		System.out.println("FPS:" + (1000.*FRAMES/(System.currentTimeMillis()-t0)));
		af.listDistances();
		af.release();
		int stride=2;
		t0=System.currentTimeMillis();
		TransformImages  ti = new TransformImages(MnistDatabase.trainImages, af,stride);
		ti.transform();
		t=System.currentTimeMillis()-t0;
		
		ResultFrame frame1 = new ResultFrame(900, 900);
		frame1.showImages(ti.getTransformedImages().subList(0, 256),16);
		System.out.println(t/1000.+" sec");
		
		t0=System.currentTimeMillis();
		TransformImages  tit = new TransformImages(MnistDatabase.testImages, af,stride);
		tit.transform();
		t=System.currentTimeMillis()-t0;
		
		ClusterImages  ci = new ClusterImages(ti.getTransformedImages(), MnistDatabase.trainLabels, 4000, 50);
		ci.cluster();
		ci.test(tit.getTransformedImages(), MnistDatabase.testLabels);
		ci.releaseOpenCl();
		
	}
}
