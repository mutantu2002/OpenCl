package home.mutant.opencl.multilayer.runners;


import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.multilayer.ArrangeFilters;
import home.mutant.opencl.multilayer.ObtainFilters;

public class MLObtainFilter {
	public static final int FRAMES = 4000;
	public static void main(String[] args) throws Exception {
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImages();
		long t0=System.currentTimeMillis();
		int noIterations=10;
		ObtainFilters  ci = new ObtainFilters(MnistDatabase.trainImages, 4, 256, noIterations);
		ci.cluster();
		long t=System.currentTimeMillis()-t0;
		System.out.println(1000.*noIterations/t+" it/sec");
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImages(ci.getClusterImages(),16);
		
		ArrangeFilters af = new ArrangeFilters(ci.getClusterImages());
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
	}
}
