package home.mutant.opencl.dot.steps;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageFloat;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryFloat;
import home.mutant.opencl.model.Program;

public class TransformImagesMapDot1D {
	List<Image> images;
	List<Image> filters;
	List<Image> transformedImages = new ArrayList<Image>();
	int imageSize;
	int filterSize;
	int transformImageSize;
	int stride;
	int dimFilter;
	int dimImageX;
	int dimImageY;
	int dimNoClusters;
	int dimTransSizeX;
	int dimTransSizeY;
	int batchItems=256*3;
	private MemoryFloat memImages;
	private MemoryFloat memFilters;
	private MemoryFloat memTransformed;
	
	private Kernel transform;
	private Program program;
	
	float[] allImages;
	float[] allFilters;
	float[] allTransformed;
	
	public TransformImagesMapDot1D(List<Image> images, List<Image> filters, int stride) {
		super();
		this.images = images;
		this.filters = filters;
		this.stride = stride;
		this.imageSize = images.get(0).getDataFloat().length;
		this.filterSize = filters.get(0).getDataFloat().length;
		this.dimImageX = images.get(0).imageX;
		this.dimImageY = images.get(0).imageY;
		this.dimFilter = (int) Math.sqrt(filterSize);
		this.dimNoClusters=(int) Math.sqrt(filters.size());
		this.dimTransSizeX=filters.size()*((dimImageX - dimFilter)/stride+1);
		this.dimTransSizeY=(dimImageY - dimFilter)/stride+1;
		this.transformImageSize=dimTransSizeX*dimTransSizeY;
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
			Image image = new ImageFloat(dimTransSizeX,dimTransSizeY);
			System.arraycopy(memTransformed.getSrc(), i*transformImageSize, image.getDataFloat(), 0, transformImageSize);
			transformedImages.add(image);
		}
	}
	
	private void prepareOpenCl(){
		Map<String, Object> params = new HashMap<>();
		params.put("IMAGE_SIZE", imageSize);
		params.put("FILTER_SIZE", filterSize);
		params.put("DIM_FILTER", dimFilter);
		params.put("NO_CLUSTERS", filters.size());
		params.put("DIM_IMAGE_X", dimImageX);
		params.put("DIM_IMAGE_Y", dimImageY);
		params.put("STRIDE", stride);

		program = new Program(Program.readResource("/dot/TransformImagesMapDot1DByCluster.c"),params);		
		
		allImages = new float[imageSize*batchItems];
		allFilters = new float[filterSize*filters.size()];
		
		for (int i=0;i<filters.size();i++){
			System.arraycopy(filters.get(i).getDataFloat(), 0, allFilters, i*(filterSize), filterSize);
		}
		allTransformed = new float[transformImageSize*batchItems];
		
		memImages = new MemoryFloat(program);
		memImages.addReadOnly(allImages);
		
		memFilters = new MemoryFloat(program);
		memFilters.addReadOnly(allFilters);

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
