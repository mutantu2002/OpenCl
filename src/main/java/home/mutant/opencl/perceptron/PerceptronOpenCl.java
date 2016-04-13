package home.mutant.opencl.perceptron;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import home.mutant.dl.models.Image;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryFloat;
import home.mutant.opencl.model.MemoryInt;
import home.mutant.opencl.model.Program;

public class PerceptronOpenCl {
	private  MemoryFloat memInputs;
	private  MemoryFloat memWeights;
	private  MemoryFloat memOutputs;
	private  MemoryInt memLabels;
	Program program;
	Kernel kernelOutput;
	
	float[] inputs;
	int[] labels;
	float[] weights;
	int noPerceptrons = 20000;
	float[] outputs;
	int batchItems=10000;
	List<Image> images;
	List<Integer> labelsList;
	int imageSize;
	
	public PerceptronOpenCl(List<Image> images, List<Integer> labelsList){
		this.images = images;
		this.labelsList = labelsList;
		
		imageSize = images.get(0).getDataFloat().length;
		inputs = new float[imageSize*batchItems];
		this.labels = new int[batchItems];
		weights= new float[noPerceptrons*(imageSize+1)];
		for (int i=0;i<weights.length;i++){
			weights[i]=(float) (1-2*Math.random());
		}
		outputs = new float[10*noPerceptrons];
		
		Map<String, Object> params = new HashMap<>();
		params.put("INPUT_SIZE", imageSize);
		params.put("NO_INPUTS", batchItems);
		program = new Program(Program.readResource("/opencl/Perceptron.c"),params);
		
		memInputs = new MemoryFloat(program);
		memInputs.addReadOnly(inputs);
		
		memWeights = new MemoryFloat(program);
		memWeights.addReadOnly(weights);
		
		memLabels = new MemoryInt(program);
		memLabels.addReadOnly(this.labels);
		
		memOutputs = new MemoryFloat(program);
		memOutputs.addReadWrite(outputs);
		
		kernelOutput = new Kernel(program, "output");
		kernelOutput.setArgument(memInputs,0);
		kernelOutput.setArgument(memLabels,1);
		kernelOutput.setArgument(memWeights,2);
		kernelOutput.setArgument(memOutputs, 3);
	}
	
	public void output(){
		for (int batch=0 ;batch<images.size()/batchItems;batch++){
			for (int i=0;i<batchItems;i++){
				System.arraycopy(images.get(batch*batchItems+i).getDataFloat(), 0, inputs, i*imageSize, imageSize);
			}
			for (int i=0;i<batchItems;i++){
				this.labels[i]=labelsList.get(batch*batchItems+i);
			}
			memInputs.copyHtoD();
			memLabels.copyHtoD();
			kernelOutput.run(noPerceptrons, 256);
		}
		memOutputs.copyDtoH();
	}
	public void calculateEntropy(){
		List<Double> entropies = new ArrayList<>();
		for(int i=0;i<noPerceptrons;i++)
		{
			double total=0;
			for (int j=0;j<10;j++){
				total+=outputs[i*10+j];
			}
			double entropy=0;
			for (int j=0;j<10;j++){
				double d = outputs[i*10+j]/total;
				entropy-=d*Math.log10(d);
			}
			if(!Double.isNaN(entropy)){
				entropies.add(entropy);
			}
			
		}
		Collections.sort(entropies, Collections.reverseOrder());
		for (Double double1 : entropies) {
			System.out.println(double1);
		}
	}
	public void releaseOpenCl(){
		memInputs.release();
		memLabels.release();
		memWeights.release();
		memOutputs.release();
		kernelOutput.release();
		program.release();
	}
}
