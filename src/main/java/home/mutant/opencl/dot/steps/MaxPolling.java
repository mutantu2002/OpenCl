package home.mutant.opencl.dot.steps;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageShort;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryShort;
import home.mutant.opencl.model.Program;

public class MaxPolling {
	List<Image> images;
	List<Image> transformedImages = new ArrayList<Image>();
	int imageSize;
	int transformImageSize;
	int dimImage;
	int dimTransformed;
	
	private MemoryShort memImages;
	private MemoryShort memTransformed;
	
	private Kernel transform;
	private Program program;
	
	short[] allImages;
	short[] allTransformed;
	
	public MaxPolling(List<Image> images) {
		super();
		this.images = images;
		this.imageSize = images.get(0).getDataShort().length;
		this.dimImage = (int) Math.sqrt(imageSize);
		this.dimTransformed=dimImage/2;
		this.transformImageSize = dimTransformed*dimTransformed;
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
			Image image = new ImageShort(transformImageSize);
			System.arraycopy(memTransformed.getSrc(), i*transformImageSize, image.getDataShort(), 0, transformImageSize);
			transformedImages.add(image);
		}
	}
	
	private void prepareOpenCl(){
		Map<String, Object> params = new HashMap<>();
		params.put("IN_IMAGE_SIZE", imageSize);
		params.put("IN_DIM_IMAGE", dimImage);
		params.put("OUT_IMAGE_SIZE", transformImageSize);
		params.put("OUT_DIM_IMAGE", dimTransformed);
		
		program = new Program(Program.readResource("/dot/MaxPolling.c"),params);		
		
		allImages = new short[imageSize*images.size()];
		
		for (int i=0;i<images.size();i++){
			System.arraycopy(images.get(i).getDataShort(), 0, allImages, i*(imageSize), imageSize);
		}
		allTransformed = new short[transformImageSize*images.size()];
		
		memImages = new MemoryShort(program);
		memImages.addReadOnly(allImages);

		memTransformed = new MemoryShort(program);
		memTransformed.addReadWrite(allTransformed);
		
		transform = new Kernel(program, "maxPolling");
		transform.setArgument(memImages,0);
		transform.setArgument(memTransformed,1);
	}
	public void releaseOpenCl(){
		memImages.release();
		memTransformed.release();
		transform.release();
		program.release();
	}
	public List<Image> getTransformedImages() {
		return transformedImages;
	}
}
