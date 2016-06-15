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

public class SoftMaxShort {
	public  double learningRate = 0.0002;
	List<Image> images;
	List<Integer> labels;
	List<Image> testImages;
	List<Integer> testLabels;
	
	List<Image> perceptronImages = new ArrayList<>();
	public List<Image> getPerceptronImages() {
		return perceptronImages;
	}
	int imageSize;
	int noIterations=10;
	int noClasses;
	int batchItems;
	int noBatches = 256;
	short[] inputImages;
	float[] perceptrons;
	float[] perceptronsUpdates;
	short[] inputLabels;
	
	short[] inputTestImages;
	short[] inputTestLabels;
	
	Program program;
	
	MemoryFloat memPerceptrons;
	MemoryShort memImages;
	MemoryFloat memUpdates;
	MemoryShort memLabels;
	Kernel train;
	
	MemoryShort memTestImages;
	MemoryShort memTestLabels;
	Kernel test;
	
	public SoftMaxShort(List<Image> images, List<Integer> labels, int noClasses) {
		super();
		this.images = images;
		this.labels = labels;
		this.noClasses = noClasses;
		this.batchItems = images.size()/noBatches;
	}
	public SoftMaxShort setTestImages(List<Image> testImages, List<Integer> testLabels){
		this.testImages = testImages;
		this.testLabels = testLabels;
		return this;
	}
	public SoftMaxShort build(){
		this.imageSize = images.get(0).getDataShort().length;
		inputImages= new short[imageSize*images.size()];
		for (int i = 0; i < images.size(); i++) {
			System.arraycopy(images.get(i).getDataShort(), 0, inputImages, i*imageSize, imageSize);
		}
		inputLabels = new short[labels.size()];
		for (int i = 0; i < inputLabels.length; i++) {
			inputLabels[i]=labels.get(i).shortValue();
		}
		perceptrons = new float[(imageSize+1)*noClasses];
		randomizePerceptrons();
		perceptronsUpdates = new float[(imageSize+1)*noClasses*noBatches];
		
		inputTestImages= new short[imageSize*testImages.size()];
		for (int i = 0; i < testImages.size(); i++) {
			System.arraycopy(testImages.get(i).getDataShort(), 0, inputTestImages, i*imageSize, imageSize);
		}
		inputTestLabels = new short[testLabels.size()];
		for (int i = 0; i < inputTestLabels.length; i++) {
			inputTestLabels[i]=testLabels.get(i).shortValue();
		}
		
		return this;
	}

	public void cluster(){
		prepareOpenCl();
		for (int iteration=0;iteration<noIterations;iteration++){
			train.run(noBatches, noBatches);
			program.finish();
	
			//System.out.println(iteration);
			memUpdates.copyDtoH();
			reducePerceptrons();
			memPerceptrons.copyHtoD();
			Arrays.fill(perceptronsUpdates, 0);
			memUpdates.copyHtoD();
			System.out.println(test());
			learningRate*=0.999;
		}
		constructPerceptronImages();
		releaseOpenCl();
	}

	private void reducePerceptrons() {
		for(int offset=0;offset<(imageSize+1)*noClasses;offset++){
			for(int batch=0;batch<noBatches;batch++){
				perceptrons[offset]+=perceptronsUpdates[batch*(imageSize+1)*noClasses+offset]*learningRate;
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
		params.put("NO_CLASSES", noClasses);
		params.put("BATCH_ITEMS", batchItems);
	
		program = new Program(Program.readResource("/dot/SoftMaxShort.c"),params);
		
		memPerceptrons = new MemoryFloat(program);
		memPerceptrons.addReadWrite(perceptrons);
		
		memImages = new MemoryShort(program);
		memImages.addReadOnly(inputImages);
		
		memUpdates = new MemoryFloat(program);
		memUpdates.addReadWrite(perceptronsUpdates);
		
		memLabels = new MemoryShort(program);
		memLabels.addReadOnly(inputLabels);
		
		train = new Kernel(program, "train");
		train.setArgument(memPerceptrons,0);
		train.setArgument(memImages,1);
		train.setArgument(memUpdates,2);
		train.setArgument(memLabels,3);
		
		memTestImages = new MemoryShort(program);
		memTestImages.addReadOnly(inputTestImages);
		
		memTestLabels = new MemoryShort(program);
		memTestLabels.addReadOnly(inputTestLabels);
		
		test = new Kernel(program, "test");
		test.setArgument(memPerceptrons,0);
		test.setArgument(memTestImages,1);
		test.setArgument(memTestLabels,2);
	}
	public double test(){
		test.run(testImages.size(), noBatches);
		program.finish();
		memTestLabels.copyDtoH();
		int count=0;
		for (int i = 0; i < inputTestLabels.length; i++) {
			if(inputTestLabels[i]==testLabels.get(i).shortValue())count++;
		}
		return ((double)count*100./testLabels.size());
	}
	
	public void releaseOpenCl(){
		memPerceptrons.release();
		memImages.release();
		memUpdates.release();
		memLabels.release();
		memTestImages.release();
		memTestLabels.release();
		train.release();
		test.release();
		program.release();
	}
	private void constructPerceptronImages(){
		for (int i=0;i<noClasses;i++) {
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
	public SoftMaxShort setNoIterations(int noIterations) {
		this.noIterations = noIterations;
		return this;
	}
}
