package home.mutant.opencl.multilayer;

import java.util.List;

import home.mutant.dl.models.Image;
import home.mutant.opencl.multilayer.steps.ArrangeFilters;
import home.mutant.opencl.multilayer.steps.ObtainFilters;
import home.mutant.opencl.multilayer.steps.TransformImages;

public class OneLayer {
	List<Image> inImages;
	List<Image> outImages;
	int frames = 100;
	int strideTransform=2;
	int strideFilters=1;
	int dimFilter=4;
	int scaleDistances=4;
	int noFilters = 256;
	ArrangeFilters af;
	
	public OneLayer(List<Image> inImages) {
		super();
		this.inImages = inImages;
	}
	public OneLayer(List<Image> inImages, int strideTransform, int strideFilters) {
		super();
		this.inImages = inImages;
		this.strideTransform = strideTransform;
		this.strideFilters = strideFilters;
	}
	public OneLayer(List<Image> inImages, int strideTransform, int strideFilters, int dimFilter, int scaleDistances) {
		super();
		this.inImages = inImages;
		this.strideTransform = strideTransform;
		this.strideFilters = strideFilters;
		this.dimFilter = dimFilter;
		this.scaleDistances = scaleDistances;
	}
	public OneLayer(List<Image> inImages, int noFilters, int strideTransform, int strideFilters, int dimFilter, int scaleDistances) {
		super();
		this.inImages = inImages;
		this.strideTransform = strideTransform;
		this.strideFilters = strideFilters;
		this.dimFilter = dimFilter;
		this.scaleDistances = scaleDistances;
		this.noFilters = noFilters;
	}
	public void transform(){
		System.out.println("Obtain filters...");
		ObtainFilters  of = new ObtainFilters(inImages, dimFilter, noFilters, 40, strideFilters);
		of.cluster();
		
		System.out.println("Arrange filters...");
		af = new ArrangeFilters(of.getClusterImages(),scaleDistances);
		long t0 = System.currentTimeMillis();
//		for(int i=0;i<frames;i++){
//			af.stepV();
//			af.show();
//			System.out.println(af.getMediumV());
//		}
		double v=10;
		while(v>2){
			af.stepV();
			af.show();
			v=af.getMediumV();
			System.out.println(v);
		}
		System.out.println("FPS:" + (1000.*frames/(System.currentTimeMillis()-t0)));
		af.copyDtoH();
		af.release();
		
		System.out.println("Transform images...");
		TransformImages  ti = new TransformImages(inImages, af,strideTransform);
		ti.transform();
		outImages = ti.getTransformedImages();
		
	}
	public List<Image> transform(List<Image> testImages){
		TransformImages  ti = new TransformImages(testImages, af,strideTransform);
		ti.transform();
		return ti.getTransformedImages();
	}
	public List<Image> getOutImages() {
		return outImages;
	}
	public List<Image> getFilters() {
		return af.images;
	}
	
}
