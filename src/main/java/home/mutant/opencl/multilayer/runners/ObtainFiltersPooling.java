package home.mutant.opencl.multilayer.runners;

import java.util.List;

import home.mutant.dl.models.Image;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.multilayer.steps.ArrangeFilters2D;
import home.mutant.opencl.multilayer.steps.ObtainFilters;

public class ObtainFiltersPooling {

 	public static void main(String[] args) throws Exception {
 		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
 		MnistDatabase.loadImages();

		long t0=System.currentTimeMillis();
		int noIterations=50;
		ObtainFilters  of = new ObtainFilters(MnistDatabase.trainImages.subList(0, 256*234), 5, 36, noIterations, 1, true);
		of.cluster();
		long t=System.currentTimeMillis()-t0;
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImages(of.getClusterImages(),6);
		System.out.println(1000.*noIterations/t+" it/sec");
		
		System.out.println("Arrange filters...");
		ArrangeFilters2D af = new ArrangeFilters2D(of.getClusterImages(),4);
		t0 = System.currentTimeMillis();

		double v=10;
		int frames=0;
		while(v>0.1){
			af.stepV();
			af.show();
			v=af.getMediumV();
			frames++;
			System.out.println(v);
		}
		System.out.println("FPS:" + (1000.*frames/(System.currentTimeMillis()-t0)));
		af.copyDtoH();
		List<Image> arrangedImages = af.getArrangedImages();
		ResultFrame frame2 = new ResultFrame(1600, 800);
		frame2.showImages(arrangedImages,6);	

		af.release();
 	}
}
