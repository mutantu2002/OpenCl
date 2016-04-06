package home.mutant.opencl.particlefilter.steps;

import java.util.HashMap;
import java.util.Map;

import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryDouble;
import home.mutant.opencl.model.MemoryFloat;
import home.mutant.opencl.model.MemoryInt;
import home.mutant.opencl.model.Program;

public class ParticleFilterOpenCl {
	float[] map;
	int[] x;
	int[] y;
	double[] weights;
	int[] rnd;
	private MemoryFloat memMap;
	private MemoryInt memX;
	private MemoryInt memY;
	private MemoryDouble memWeights;
	private MemoryInt memRnd;
	
	private Kernel estimate;
	private Kernel move;
	private Program program;
	
	public ParticleFilterOpenCl(float[] map, int[] x, int[] y, double[] weights, int[] rnd) {
		super();
		this.map = map;
		this.x = x;
		this.y = y;
		this.weights = weights;
		this.rnd=rnd;
		prepareOpenCl();
	}
	private void prepareOpenCl(){
		Map<String, Object> params = new HashMap<>();
		params.put("DIM_MAP", ((int)Math.sqrt(map.length)));
		params.put("MEAN", 20);
		params.put("NO_RND", rnd.length);
		
		program = new Program(Program.readResource("/opencl/ParticleFilter.c"),params);		
		
		memMap = new MemoryFloat(program);
		memMap.addReadOnly(map);

		memX = new MemoryInt(program);
		memX.addReadWrite(x);
		memY = new MemoryInt(program);
		memY.addReadWrite(y);
		
		memWeights = new MemoryDouble(program);
		memWeights.addReadWrite(weights);
		
		memRnd = new MemoryInt(program);
		memRnd.addReadOnly(rnd);
		
		estimate = new Kernel(program, "estimateWeights");
		estimate.setArgument(memMap,0);
		estimate.setArgument(memX,1);
		estimate.setArgument(memY,2);
		estimate.setArgument(memWeights,3);
		
		move = new Kernel(program, "moveParticles");
		move.setArgument(memX,0);
		move.setArgument(memY,1);
		move.setArgument(memRnd,2);
	}
	public void estimate(float measurement, int noParticles){
		memX.copyHtoD();
		memY.copyHtoD();
		estimate.setArgument(measurement,4);
		estimate.run(noParticles, 256);
		program.finish();
		memWeights.copyDtoH();
	}
	public void move(int dx, int dy, int indexRnd, int noParticles){
		memX.copyHtoD();
		memY.copyHtoD();
		move.setArgument(indexRnd,3);
		move.setArgument(dx,4);
		move.setArgument(dy,5);
		move.run(noParticles, 256);
		memX.copyDtoH();
		memY.copyDtoH();
	}
	public void releaseOpenCl(){
		memMap.release();
		memX.release();
		memY.release();
		memWeights.release();
		memRnd.release();
		estimate.release();
		move.release();
		program.release();
	}
}
