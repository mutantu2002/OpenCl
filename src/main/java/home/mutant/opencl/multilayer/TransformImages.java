package home.mutant.opencl.multilayer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageDouble;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryDouble;
import home.mutant.opencl.model.Program;
import home.mutant.opencl.smooth.fourd.LinkedClusterablesOpenCl4D;

public class TransformImages {
	List<Image> images;
	LinkedClusterablesOpenCl4D filters;
	List<Image> transformedImages = new ArrayList<Image>();
	int imageSize;
	int filterSize;
	int transformImageSize;
	int stride;
	int dimFilter;
	int dimImage;
	
	private MemoryDouble memX;
	private MemoryDouble memY;
	private MemoryDouble memZ;
	private MemoryDouble memW;
	
	private MemoryDouble memImages;
	private MemoryDouble memFilters;
	private MemoryDouble memTransformed;
	
	private Kernel transform;
	private Program program;
	
	double[] allImages;
	double[] allFilters;
	double[] allTransformed;
	
	public TransformImages(List<Image> images, LinkedClusterablesOpenCl4D filters, int stride) {
		super();
		this.images = images;
		this.filters = filters;
		this.stride = stride;
		this.imageSize = images.get(0).getDataDouble().length;
		this.filterSize = filters.filters.clusterables.get(0).getWeights().length;
		this.dimImage = (int) Math.sqrt(imageSize);
		this.dimFilter = (int) Math.sqrt(filterSize);
		int dimTransSize=2*((dimImage - dimFilter)/stride+1);
		this.transformImageSize=dimTransSize*dimTransSize;
	}
	public void transform(){
		prepareOpenCl();
		transform.run(images.size(), 256);
		program.finish();
		memTransformed.copyDtoH();
		contructTransformedImages();
		releaseOpenCl();
	}
	private void contructTransformedImages(){
		for (int i=0;i<images.size();i++) {
			Image image = new ImageDouble(transformImageSize);
			System.arraycopy(memTransformed.getSrc(), i*transformImageSize, image.getDataDouble(), 0, transformImageSize);
			transformedImages.add(image);
		}
	}
	
	private void prepareOpenCl(){
		Map<String, Object> params = new HashMap<>();
		params.put("IMAGE_SIZE", imageSize);
		params.put("FILTER_SIZE", filterSize);
		params.put("NO_CLUSTERS", filters.filters.clusterables.size());
		params.put("DIM_FILTER", dimFilter);
		params.put("DIM_IMAGE", dimImage);
		params.put("STRIDE", stride);
		program = new Program(Program.readResource("/opencl/TransformImages4D.c"),params);		
		
		allImages = new double[imageSize*images.size()];
		allFilters = new double[filterSize*filters.filters.clusterables.size()];
		
		for (int i=0;i<images.size();i++){
			System.arraycopy(images.get(i).getDataDouble(), 0, allImages, i*(imageSize), imageSize);
		}
		for (int i=0;i<filters.filters.clusterables.size();i++){
			System.arraycopy(filters.filters.clusterables.get(i).getWeights(), 0, allFilters, i*(filterSize), filterSize);
		}
		allTransformed = new double[transformImageSize*images.size()];
		
		memImages = new MemoryDouble(program);
		memImages.addReadOnly(allImages);
		
		memFilters = new MemoryDouble(program);
		memFilters.addReadWrite(allFilters);

		memTransformed = new MemoryDouble(program);
		memTransformed.addReadWrite(allTransformed);
		
		memX = new MemoryDouble(program);
		memX.addReadWrite(filters.x);
		
		memY = new MemoryDouble(program);
		memY.addReadWrite(filters.y);
		
		memZ = new MemoryDouble(program);
		memZ.addReadWrite(filters.z);
		
		memW = new MemoryDouble(program);
		memW.addReadWrite(filters.w);
		
		transform = new Kernel(program, "transform");
		transform.setArguments(memImages,memFilters,memX,memY,memZ,memW,memTransformed);
	}
	public void releaseOpenCl(){
		memImages.release();
		memFilters.release();
		memX.release();
		memY.release();
		memZ.release();
		memW.release();
		memTransformed.release();
		transform.release();
		program.release();
	}
	public List<Image> getTransformedImages() {
		return transformedImages;
	}
}
