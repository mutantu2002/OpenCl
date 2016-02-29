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

public class MeanPollingImages {
	List<Image> images;
	List<Image> transformedImages = new ArrayList<Image>();
	int imageSize;
	int transformImageSize;
	int dimImage;
	int dimTransformed;
	
	private MemoryFloat memImages;
	private MemoryFloat memTransformed;
	
	private Kernel transform;
	private Program program;
	
	float[] allImages;
	float[] allTransformed;
	
	public MeanPollingImages(List<Image> images) {
		super();
		this.images = images;
		this.imageSize = images.get(0).getDataFloat().length;
		this.dimImage = (int) Math.sqrt(imageSize);
		this.dimTransformed=(dimImage/4)*2;
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
			Image image = new ImageFloat(transformImageSize);
			System.arraycopy(memTransformed.getSrc(), i*transformImageSize, image.getDataFloat(), 0, transformImageSize);
			transformedImages.add(image);
		}
	}
	
	private void prepareOpenCl(){
		Map<String, Object> params = new HashMap<>();
		params.put("IN_IMAGE_SIZE", imageSize);
		params.put("IN_DIM_IMAGE", dimImage);
		params.put("OUT_IMAGE_SIZE", transformImageSize);
		params.put("OUT_DIM_IMAGE", dimTransformed);
		
		program = new Program(Program.readResource("/opencl/MeanPolling4D.c"),params);		
		
		allImages = new float[imageSize*images.size()];
		
		for (int i=0;i<images.size();i++){
			System.arraycopy(images.get(i).getDataFloat(), 0, allImages, i*(imageSize), imageSize);
		}
		allTransformed = new float[transformImageSize*images.size()];
		
		memImages = new MemoryFloat(program);
		memImages.addReadOnly(allImages);

		memTransformed = new MemoryFloat(program);
		memTransformed.addReadWrite(allTransformed);
		
		transform = new Kernel(program, "meanPolling4D");
		transform.setArguments(memImages,memTransformed);
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
