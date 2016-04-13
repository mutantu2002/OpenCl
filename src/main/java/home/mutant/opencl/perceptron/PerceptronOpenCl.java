package home.mutant.opencl.perceptron;

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
	int noPerceptrons = 200;
	float[] outputs;
	public PerceptronOpenCl(List<Image> images, List<Integer> labels){
		int imageSize = images.get(0).getDataFloat().length;
		inputs = new float[imageSize*images.size()];
		this.labels = new int[labels.size()];
		for (int i=0;i<images.size();i++){
			System.arraycopy(images.get(i).getDataFloat(), 0, inputs, i*(imageSize), imageSize);
		}
		for (int i=0;i<labels.size();i++){
			this.labels[i]=labels.get(i);
		}
		weights= new float[noPerceptrons*(imageSize+1)];
		for (int i=0;i<weights.length;i++){
			weights[i]=(float) (1-2*Math.random());
		}
		outputs = new float[10*noPerceptrons];
		
		Map<String, Object> params = new HashMap<>();
		params.put("INPUT_SIZE", imageSize);
		params.put("NO_INPUTS", images.size());
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
		kernelOutput.run(noPerceptrons, 200);
		memOutputs.copyDtoH();
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
