package home.mutant.opencl.dot.steps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageFloat;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryFloat;
import home.mutant.opencl.model.MemoryShort;
import home.mutant.opencl.model.Program;

public class GradientClustering {
	public  double learningRate = 0.008;
	List<Image> images;
	
	List<Image> perceptronImages = new ArrayList<>();
	public List<Image> getPerceptronImages() {
		return perceptronImages;
	}
	int imageSize;
	int noIterations=10;
	int noPerceptrons;
	int batchItems;
	int noBatches = 256;
	short[] inputImages;
	float[] perceptrons;
	float[] perceptronsUpdates;
	
	float[] activations;
	
	Program program;
	
	MemoryFloat memPerceptrons;
	MemoryShort memImages;
	MemoryFloat memUpdates;
	Kernel train;
	
	MemoryFloat memActivations;
	Kernel forward;
	
	public GradientClustering(List<Image> images, int noPerceptrons) {
		super();
		this.images = images;
		this.noPerceptrons = noPerceptrons;
		this.batchItems = images.size()/noBatches;
	}

	public GradientClustering build(){
		this.imageSize = images.get(0).getDataShort().length;
		inputImages= new short[imageSize*images.size()];
		for (int i = 0; i < images.size(); i++) {
			System.arraycopy(images.get(i).getDataShort(), 0, inputImages, i*imageSize, imageSize);
		}

		perceptrons = new float[imageSize*noPerceptrons];
		randomizePerceptrons();
		perceptronsUpdates = new float[imageSize*noPerceptrons*noBatches];
		
		activations = new float[images.size()*noPerceptrons];
		return this;
	}
	public double test(){
		forward.run(images.size(), noBatches);
		program.finish();
		memActivations.copyDtoH();
		double d=0;
		for (int im=0;im<images.size()-1;im++){
			for(int p=0;p<noPerceptrons;p++){
				d+=(activations[im*noPerceptrons+p]-activations[(im+1)*noPerceptrons+p])*(activations[im*noPerceptrons+p]-activations[(im+1)*noPerceptrons+p]);
			}
		}
		return d;

	}
	public void cluster(){
		prepareOpenCl();
		for (int iteration=0;iteration<noIterations;iteration++){
			System.out.println(test());
			train.run(noBatches, noBatches);
			program.finish();
	
			//System.out.println(iteration);
			memUpdates.copyDtoH();
			reducePerceptrons();
			memPerceptrons.copyHtoD();
			Arrays.fill(perceptronsUpdates, 0);
			memUpdates.copyHtoD();
			//learningRate*=0.999;
		}
		constructPerceptronImages();
		releaseOpenCl();
	}

	private void reducePerceptrons() {
		for(int offset=0;offset<imageSize*noPerceptrons;offset++){
			for(int batch=0;batch<noBatches;batch++){
				perceptrons[offset]+=perceptronsUpdates[batch*imageSize*noPerceptrons+offset]*learningRate;
			}
		}
	}

	private void randomizePerceptrons() {
		for (int i = 0; i < perceptrons.length; i++) {
			perceptrons[i] = (float) (1-2*Math.random());
		}
	}

	private void prepareOpenCl(){
		Map<String, Object> params = new HashMap<>();
		params.put("IMAGE_SIZE", imageSize);
		params.put("NO_PERCEPTRONS", noPerceptrons);
		params.put("BATCH_ITEMS", batchItems);
	
		program = new Program(Program.readResource("/dot/GradientClustering.c"),params);
		
		memPerceptrons = new MemoryFloat(program);
		memPerceptrons.addReadWrite(perceptrons);
		
		memImages = new MemoryShort(program);
		memImages.addReadOnly(inputImages);
		
		memUpdates = new MemoryFloat(program);
		memUpdates.addReadWrite(perceptronsUpdates);
		
		train = new Kernel(program, "train");
		train.setArgument(memPerceptrons,0);
		train.setArgument(memImages,1);
		train.setArgument(memUpdates,2);
		
		memActivations = new MemoryFloat(program);
		memActivations.addReadWrite(activations);
		
		forward = new Kernel(program, "forward");
		forward.setArgument(memPerceptrons,0);
		forward.setArgument(memImages,1);
		forward.setArgument(memActivations,2);
	}
	
	public void releaseOpenCl(){
		memPerceptrons.release();
		memImages.release();
		memUpdates.release();
		memActivations.release();
		train.release();
		forward.release();
		program.release();
	}
	private void constructPerceptronImages(){
		for (int i=0;i<noPerceptrons;i++) {
			Image image = new ImageFloat(imageSize);
			double max = -1*Double.MAX_VALUE;
			double min = Double.MAX_VALUE;
			int clusterOffset = (imageSize+1)*i;
			for (int j = 0; j < imageSize; j++) {
				if (perceptrons[clusterOffset+j]>max)max=perceptrons[clusterOffset+j];
				if (perceptrons[clusterOffset+j]<min)min=perceptrons[clusterOffset+j];
			}
			max=255/(max-min);
			for (int j = 0; j < imageSize; j++) {
				image.getDataFloat()[j]=(float) ((perceptrons[clusterOffset+j]-min)*max);
			}
			perceptronImages.add(image);
		}			
	}
	public GradientClustering setNoIterations(int noIterations) {
		this.noIterations = noIterations;
		return this;
	}
}
