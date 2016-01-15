package home.mutant.opencl.smooth.threed;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.kmeans.model.Clusterable;
import home.mutant.dl.utils.kmeans.model.ListClusterable;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryDouble;
import home.mutant.opencl.model.Program;

public class LinkedClusterablesOpenCl3D implements Serializable{
	private static final long serialVersionUID = -6967038438910323276L;
	public transient double preDistances[];
	public ListClusterable filters;
	public double[] x;
	public double[] y;
	public double[] z;
	transient double[] vx;
	transient double[] vy;
	transient double[] vz;
	transient double dt=0.0002;
	transient double K=1;
	transient double friction=0.1;
	
	transient ResultFrame frame;
	private transient MemoryDouble memX;
	private transient MemoryDouble memY;
	private transient MemoryDouble memZ;
	private transient MemoryDouble memVx;
	private transient MemoryDouble memVy;
	private transient MemoryDouble memVz;
	private transient MemoryDouble memPredistances;
	private transient Kernel stepV;
	private transient Program program;
	
	public LinkedClusterablesOpenCl3D(ListClusterable clusterables) {
		super();
		this.filters = clusterables;
		fillPreDistances();
		x = new double[filters.clusterables.size()];
		y = new double[filters.clusterables.size()];
		z = new double[filters.clusterables.size()];
		vx = new double[filters.clusterables.size()];
		vy = new double[filters.clusterables.size()];
		vz = new double[filters.clusterables.size()];
		frame = new ResultFrame(800, 800);
		randDistances();
		
		Map<String, Object> params = new HashMap<>();
		params.put("NO_PARTICLES", filters.clusterables.size());
		params.put("DT", dt);
		params.put("K", K);
		params.put("FRICTION", friction);
		program = new Program(Program.readResource("/opencl/ElasticSmoothie3D.c"),params);
		
		memX = new MemoryDouble(program);
		memX.addReadWrite(x);
		
		memY = new MemoryDouble(program);
		memY.addReadWrite(y);
		
		memZ = new MemoryDouble(program);
		memZ.addReadWrite(z);
		
		memVx = new MemoryDouble(program);
		memVx.addReadWrite(vx);
		
		memVy = new MemoryDouble(program);
		memVy.addReadWrite(vy);
		
		memVz = new MemoryDouble(program);
		memVz.addReadWrite(vz);
		
		memPredistances = new MemoryDouble(program);
		memPredistances.addReadOnly(preDistances);
		
		stepV = new Kernel(program, "stepV");
		stepV.setArguments(memX,memY,memZ,memVx, memVy,memVz, memPredistances);
		
	}
	private  void fillPreDistances(){
		int size = filters.clusterables.size();
		preDistances = new double[size*size];
		for (int i=0;i<size;i++){
			for (int j=0;j<size;j++){
				preDistances[i*size+j]=filters.clusterables.get(i).d(filters.clusterables.get(j))/5;
			}
		}
	}
	private void randDistances(){
		for (int i=0;i<filters.clusterables.size();i++){
			x[i]=Math.random()*200-100;
			y[i]=Math.random()*200-100;
			z[i]=Math.random()*200-100;
		}
	}
	public void show(){
		copyDtoH();
		frame.drawingPanel.empty();
		for (int i=0;i<filters.clusterables.size();i++){
			Clusterable clusterable = filters.clusterables.get(i);
			int x1=(int) (x[i]+400);
			int y1=(int) (y[i]+400);
			if(x1>=0 && x1<796 && y1>=0 && y1<796){
				frame.putImage(clusterable.getImage(), x1, y1);
			}
		}
		frame.repaint();
	}
	public void showFilters(){
		filters.show();
	}

	public void stepV(){
		stepV.run(filters.clusterables.size(), 256);
		program.finish();
	}
	
	public void listDistances(){
		copyDtoH();
		double error=0;
		memPredistances.copyDtoH();
		int size = filters.clusterables.size();
		for (int i=0;i<size;i++){
			for (int j=0;j<size;j++){
				if(i==j) continue;
				double postDist = Math.sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j])+(z[i]-z[j])*(z[i]-z[j]));
				System.out.println(postDist+" - "+preDistances[i*size+j]);
				error+=Math.abs(postDist-preDistances[i*size+j])/preDistances[i*size+j];
			}
		}
		System.out.println("Error "+error);
	}
	private void copyDtoH(){
		memX.copyDtoH();
		memY.copyDtoH();
		memZ.copyDtoH();
		memVx.copyDtoH();
		memVy.copyDtoH();
		memVz.copyDtoH();
	}
	public void release(){
		memX.release();
		memY.release();
		memZ.release();
		memVx.release();
		memVy.release();
		memVz.release();
		memPredistances.release();
		stepV.release();
		program.release();
	}
}
