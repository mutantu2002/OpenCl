package home.mutant.opencl.multilayer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageFloat;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryFloat;
import home.mutant.opencl.model.Program;

public class TransformImages {
	List<Image> images;
	ArrangeFilters filters;
	List<Image> transformedImages = new ArrayList<Image>();
	int imageSize;
	int filterSize;
	int transformImageSize;
	int stride;
	int dimFilter;
	int dimImage;
	
	private MemoryFloat memX;
	private MemoryFloat memY;
	private MemoryFloat memZ;
	private MemoryFloat memW;
	
	private MemoryFloat memImages;
	private MemoryFloat memFilters;
	private MemoryFloat memTransformed;
	
	private Kernel transform;
	private Program program;
	
	float[] allImages;
	float[] allFilters;
	float[] allTransformed;
	
	public TransformImages(List<Image> images, ArrangeFilters filters, int stride) {
		super();
		this.images = images;
		this.filters = filters;
		this.stride = stride;
		this.imageSize = images.get(0).getDataFloat().length;
		this.filterSize = filters.images.get(0).getDataFloat().length;
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
			Image image = new ImageFloat(transformImageSize);
			System.arraycopy(memTransformed.getSrc(), i*transformImageSize, image.getDataFloat(), 0, transformImageSize);
			transformedImages.add(image);
		}
	}
	
	private void prepareOpenCl(){
		Map<String, Object> params = new HashMap<>();
		params.put("IMAGE_SIZE", imageSize);
		params.put("FILTER_SIZE", filterSize);
		params.put("NO_CLUSTERS", filters.images.size());
		params.put("DIM_FILTER", dimFilter);
		params.put("DIM_IMAGE", dimImage);
		params.put("STRIDE", stride);
		program = new Program(Program.readResource("/opencl/TransformImages4DFloat.c"),params);		
		
		allImages = new float[imageSize*images.size()];
		allFilters = new float[filterSize*filters.images.size()];
		
		for (int i=0;i<images.size();i++){
			System.arraycopy(images.get(i).getDataFloat(), 0, allImages, i*(imageSize), imageSize);
		}
		for (int i=0;i<filters.images.size();i++){
			System.arraycopy(filters.images.get(i).getDataFloat(), 0, allFilters, i*(filterSize), filterSize);
		}
		allTransformed = new float[transformImageSize*images.size()];
		
		memImages = new MemoryFloat(program);
		memImages.addReadOnly(allImages);
		
		memFilters = new MemoryFloat(program);
		memFilters.addReadWrite(allFilters);

		memTransformed = new MemoryFloat(program);
		memTransformed.addReadWrite(allTransformed);
		
		memX = new MemoryFloat(program);
		memX.addReadWrite(filters.x);
		
		memY = new MemoryFloat(program);
		memY.addReadWrite(filters.y);
		
		memZ = new MemoryFloat(program);
		memZ.addReadWrite(filters.z);
		
		memW = new MemoryFloat(program);
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
