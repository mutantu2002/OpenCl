package home.mutant.opencl.dot.steps;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageFloat;
import home.mutant.dl.models.ImageShort;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryFloat;
import home.mutant.opencl.model.MemoryShort;
import home.mutant.opencl.model.Program;

public class TransformImagesMapDot1DShort {
	List<Image> images;
	List<Image> filters;
	List<Image> transformedImages = new ArrayList<Image>();
	int imageSize;
	int filterSize;
	int transformImageSize;
	int strideX;
	int strideY;
	int dimFilterX;
	int dimFilterY;
	int dimImageX;
	int dimImageY;
	int dimTransSizeX;
	int dimTransSizeY;
	int batchItems=256*3;
	private MemoryShort memImages;
	private MemoryFloat memFilters;
	private MemoryShort memTransformed;
	
	private Kernel transform;
	private Program program;
	
	short[] allImages;
	float[] allFilters;
	short[] allTransformed;
	
	public TransformImagesMapDot1DShort(List<Image> images, List<Image> filters) {
		this.images = images;
		this.filters = filters;
	}
	public TransformImagesMapDot1DShort build(){
		this.imageSize = images.get(0).getDataShort().length;
		this.filterSize = filters.get(0).getDataFloat().length;
		this.dimImageX = images.get(0).imageX;
		this.dimImageY = images.get(0).imageY;
		this.dimFilterX = filters.get(0).imageX;
		this.dimFilterY = filters.get(0).imageY;
		this.dimTransSizeX=((dimImageX - dimFilterX)/strideX+1);
		this.dimTransSizeY=filters.size()*((dimImageY - dimFilterY)/strideY+1);
		this.transformImageSize=dimTransSizeX*dimTransSizeY;
		return this;
	}
	public TransformImagesMapDot1DShort setStrideX(int strideX){
		this.strideX = strideX;
		return this;
	}
	public TransformImagesMapDot1DShort setStrideY(int strideY){
		this.strideY = strideY;
		return this;
	}
	public void transform(){
		prepareOpenCl();
		for (int batch=0 ;batch<images.size()/batchItems;batch++){
			for (int i=0;i<batchItems;i++){
				System.arraycopy(images.get(batch*batchItems+i).getDataShort(), 0, allImages, i*(imageSize), imageSize);
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
			Image image = new ImageShort(dimTransSizeX,dimTransSizeY);
			double max = -1*Double.MAX_VALUE;
			double min = Double.MAX_VALUE;
			int clusterOffset = dimTransSizeX*dimTransSizeY*i;
			for (int j = 0; j < dimTransSizeX*dimTransSizeY; j++) {
				if (allTransformed[clusterOffset+j]>max)max=allTransformed[clusterOffset+j];
				if (allTransformed[clusterOffset+j]<min)min=allTransformed[clusterOffset+j];
			}
			max=255/(max-min);
			for (int j = 0; j < dimTransSizeX*dimTransSizeY; j++) {
				image.getDataShort()[j]=(short) ((allTransformed[clusterOffset+j]-min)*max);
			}
			transformedImages.add(image);
		}			
	}	
	private void prepareOpenCl(){
		Map<String, Object> params = new HashMap<>();
		params.put("IMAGE_SIZE", imageSize);
		params.put("FILTER_SIZE", filterSize);
		params.put("DIM_FILTER_X", dimFilterX);
		params.put("DIM_FILTER_Y", dimFilterY);
		params.put("NO_CLUSTERS", filters.size());
		params.put("DIM_IMAGE_X", dimImageX);
		params.put("DIM_IMAGE_Y", dimImageY);
		params.put("STRIDE_X", strideX);
		params.put("STRIDE_Y", strideY);

		program = new Program(Program.readResource("/dot/TransformImagesMapDot1DByClusterShort.c"),params);		
		
		allImages = new short[imageSize*batchItems];
		allFilters = new float[filterSize*filters.size()];
		
		for (int i=0;i<filters.size();i++){
			System.arraycopy(filters.get(i).getDataFloat(), 0, allFilters, i*(filterSize), filterSize);
		}
		allTransformed = new short[transformImageSize*batchItems];
		
		memImages = new MemoryShort(program);
		memImages.addReadOnly(allImages);
		
		memFilters = new MemoryFloat(program);
		memFilters.addReadOnly(allFilters);

		memTransformed = new MemoryShort(program);
		memTransformed.addReadWrite(allTransformed);
		
		transform = new Kernel(program, "transform");
		transform.setArgument(memImages,0);
		transform.setArgument(memFilters,1);
		transform.setArgument(memTransformed,2);
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
