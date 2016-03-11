package home.mutant.opencl.particlefilter.steps;

import java.util.HashMap;
import java.util.Map;

import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryDouble;
import home.mutant.opencl.model.MemoryFloat;
import home.mutant.opencl.model.MemoryInt;
import home.mutant.opencl.model.Program;

public class EstimateMeasurement {
	float[] map;
	int[] x;
	int[] y;
	double[] weights;
	
	private MemoryFloat memMap;
	private MemoryInt memX;
	private MemoryInt memY;
	private MemoryDouble memWeights;
	
	private Kernel estimate;
	private Program program;
	
	public EstimateMeasurement(float[] map, int[] x, int[] y, double[] weights) {
		super();
		this.map = map;
		this.x = x;
		this.y = y;
		this.weights = weights;
		prepareOpenCl();
	}
	private void prepareOpenCl(){
		Map<String, Object> params = new HashMap<>();
		params.put("DIM_MAP", ((int)Math.sqrt(map.length)));
		params.put("MEAN", 40);
		
		program = new Program(Program.readResource("/opencl/ParticleFilter.c"),params);		
		
		memMap = new MemoryFloat(program);
		memMap.addReadOnly(map);

		memX = new MemoryInt(program);
		memX.addReadOnly(x);
		memY = new MemoryInt(program);
		memY.addReadOnly(y);
		
		memWeights = new MemoryDouble(program);
		memWeights.addReadWrite(weights);
		
		estimate = new Kernel(program, "estimateWeights");
		estimate.setArgument(memMap,0);
		estimate.setArgument(memX,1);
		estimate.setArgument(memY,2);
		estimate.setArgument(memWeights,3);
	}
	public void estimate(float measurement, int noParticles){
		memX.copyHtoD();
		memY.copyHtoD();
		estimate.setArgument(measurement,4);
		estimate.run(noParticles, 256);
		program.finish();
		memWeights.copyDtoH();
	}
	public void releaseOpenCl(){
		memMap.release();
		memX.release();
		memY.release();
		memWeights.release();
		estimate.release();
		program.release();
	}
}
