package home.mutant.opencl.multilayer;

import java.util.List;

import home.mutant.dl.models.Image;

public class OneLayer {
	List<Image> inImages;
	List<Image> outImages;
	int frames = 4000;
	int strideTransform=2;
	int strideFilters=1;
	
	public OneLayer(List<Image> inImages) {
		super();
		this.inImages = inImages;
	}
	public void transform(){
		System.out.println("Obtain filters...");
		ObtainFilters  of = new ObtainFilters(inImages, 4, 256, 40, strideFilters);
		of.cluster();
		
		System.out.println("Arrange filters...");
		ArrangeFilters af = new ArrangeFilters(of.getClusterImages());
		for(int i=0;i<frames;i++){
			af.stepV();
		}
		af.release();
		
		System.out.println("Transform images...");
		TransformImages  ti = new TransformImages(inImages, af,strideTransform);
		ti.transform();
		outImages = ti.getTransformedImages();
		
	}
	public List<Image> getOutImages() {
		return outImages;
	}
	
}
