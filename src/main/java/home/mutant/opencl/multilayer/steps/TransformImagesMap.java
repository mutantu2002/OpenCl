package home.mutant.opencl.multilayer.steps;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageFloat;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryFloat;
import home.mutant.opencl.model.Program;

public class TransformImagesMap {
	List<Image> images;
	List<Image> filters;
	List<Image> transformedImages = new ArrayList<Image>();
	int imageSize;
	int filterSize;
	int transformImageSize;
	int stride;
	int dimFilter;
	int dimImage;
	int dimNoClusters;
	int batchItems=10000;
	private MemoryFloat memImages;
	private MemoryFloat memFilters;
	private MemoryFloat memTransformed;
	
	private Kernel transform;
	private Program program;
	
	float[] allImages;
	float[] allFilters;
	float[] allTransformed;
	
	public TransformImagesMap(List<Image> images, List<Image> filters, int stride) {
		super();
		this.images = images;
		this.filters = filters;
		this.stride = stride;
		this.imageSize = images.get(0).getDataFloat().length;
		this.filterSize = filters.get(0).getDataFloat().length;
		this.dimImage = (int) Math.sqrt(imageSize);
		this.dimFilter = (int) Math.sqrt(filterSize);
		this.dimNoClusters=(int) Math.sqrt(filters.size());
		int dimTransSize=dimNoClusters*((dimImage - dimFilter)/stride+1);
		this.transformImageSize=dimTransSize*dimTransSize;
		System.out.println(dimTransSize);
	}
	public void transform(){
		prepareOpenCl();
		for (int batch=0 ;batch<images.size()/batchItems;batch++){
			for (int i=0;i<batchItems;i++){
				System.arraycopy(images.get(batch*batchItems+i).getDataFloat(), 0, allImages, i*(imageSize), imageSize);
			}
			memImages.copyHtoD();
			transform.run(batchItems, 256);
			program.finish();
			memTransformed.copyDtoH();
			contructTransformedImages();
		}
		releaseOpenCl();
	}
	private void contructTransformedImages(){
		for (int i=0;i<batchItems;i++) {
			Image image = new ImageFloat(transformImageSize);
			System.arraycopy(memTransformed.getSrc(), i*transformImageSize, image.getDataFloat(), 0, transformImageSize);
			transformedImages.add(image);
		}
	}
	
	private void prepareOpenCl(){
		Map<String, Object> params = new HashMap<>();
		params.put("IMAGE_SIZE", imageSize);
		params.put("FILTER_SIZE", filterSize);
		params.put("DIM_FILTER", dimFilter);
		params.put("DIM_NO_CLUSTERS", dimNoClusters);
		params.put("DIM_IMAGE", dimImage);
		params.put("STRIDE", stride);
		params.put("MEAN", 300);
		program = new Program(Program.readResource("/opencl/TransformImagesMap.c"),params);		
		
		allImages = new float[imageSize*batchItems];
		allFilters = new float[filterSize*filters.size()];
		
		for (int i=0;i<filters.size();i++){
			System.arraycopy(filters.get(i).getDataFloat(), 0, allFilters, i*(filterSize), filterSize);
		}
		allTransformed = new float[transformImageSize*batchItems];
		
		memImages = new MemoryFloat(program);
		memImages.addReadOnly(allImages);
		
		memFilters = new MemoryFloat(program);
		memFilters.addReadWrite(allFilters);

		memTransformed = new MemoryFloat(program);
		memTransformed.addReadWrite(allTransformed);
		
		transform = new Kernel(program, "transform");
		transform.setArguments(memImages,memFilters,memTransformed);
	}
	public void releaseOpenCl(){
		memImages.release();
		memFilters.release();
		memTransformed.release();
		transform.release();
		program.release();
	}
	public List<Image> getTransformedImages() {
		return transformedImages;
	}
}
