package home.mutant.opencl.multilayer.steps;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import home.mutant.dl.models.Image;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryFloat;
import home.mutant.opencl.model.Program;

public class ArrangeFilters {
	public List<Image> images;
	
	public  float preDistances[];
	public float[] x;
	public float[] y;
	public float[] z;
	public float[] w;
	float[] vx;
	float[] vy;
	float[] vz;
	float[] vw;
	float dt=0.0008f;
	float K=1;
	float friction=0.1f;
	
	ResultFrame frame;
	private  MemoryFloat memX;
	private  MemoryFloat memY;
	private  MemoryFloat memZ;
	private  MemoryFloat memW;
	private  MemoryFloat memVx;
	private  MemoryFloat memVy;
	private  MemoryFloat memVz;
	private  MemoryFloat memVw;
	private  MemoryFloat memPredistances;
	private  Kernel stepV;
	private  Program program;
	
	public ArrangeFilters(List<Image> images) {
		super();
		this.images = images;
		fillPreDistances();
		x = new float[images.size()];
		y = new float[images.size()];
		z = new float[images.size()];
		w = new float[images.size()];
		vx = new float[images.size()];
		vy = new float[images.size()];
		vz = new float[images.size()];
		vw = new float[images.size()];
		frame = new ResultFrame(800, 800);
		randDistances();
		
		Map<String, Object> params = new HashMap<>();
		params.put("NO_PARTICLES", images.size());
		params.put("DT", dt);
		params.put("K", K);
		params.put("FRICTION", friction);
		program = new Program(Program.readResource("/opencl/ElasticSmoothie4DFloat.c"),params);
		
		memX = new MemoryFloat(program);
		memX.addReadWrite(x);
		
		memY = new MemoryFloat(program);
		memY.addReadWrite(y);
		
		memZ = new MemoryFloat(program);
		memZ.addReadWrite(z);
		
		memW = new MemoryFloat(program);
		memW.addReadWrite(w);
		
		memVx = new MemoryFloat(program);
		memVx.addReadWrite(vx);
		
		memVy = new MemoryFloat(program);
		memVy.addReadWrite(vy);
		
		memVz = new MemoryFloat(program);
		memVz.addReadWrite(vz);
	
		memVw = new MemoryFloat(program);
		memVw.addReadWrite(vw);
		
		memPredistances = new MemoryFloat(program);
		memPredistances.addReadOnly(preDistances);
		
		stepV = new Kernel(program, "stepV");
		stepV.setArguments(memX,memY,memZ,memW,memVx, memVy,memVz,memVw, memPredistances);
		
	}
	private  void fillPreDistances(){
		int size = images.size();
		preDistances = new float[size*size];
		for (int i=0;i<size;i++){
			for (int j=0;j<size;j++){
				preDistances[i*size+j]=d(images.get(i),images.get(j))/5;
			}
		}
	}
	
	private float d(Image i1, Image i2){
		double d=0;
		for (int i = 0; i < i1.getDataFloat().length; i++) {
			d+=(i1.getDataFloat()[i] - i2.getDataFloat()[i])*(i1.getDataFloat()[i] - i2.getDataFloat()[i]);
		}
		return (float) Math.sqrt(d);
	}
	private void randDistances(){
		for (int i=0;i<images.size();i++){
			x[i]=(float) (Math.random()*200-100);
			y[i]=(float) (Math.random()*200-100);
			z[i]=(float) (Math.random()*200-100);
			w[i]=(float) (Math.random()*200-100);
		}
	}
	public void show(){
		copyDtoH();
		frame.drawingPanel.empty();
		for (int i=0;i<images.size();i++){
			Image image = images.get(i);
			int x1=(int) (x[i]+400);
			int y1=(int) (y[i]+400);
			if(x1>=0 && x1<796 && y1>=0 && y1<796){
				frame.putImage(image, x1, y1);
			}
		}
		frame.repaint();
	}

	public void stepV(){
		stepV.run(images.size(), images.size());
		program.finish();
	}
	
	public void listDistances(){
		copyDtoH();
		double error=0;
		memPredistances.copyDtoH();
		int size = images.size();
		for (int i=0;i<size;i++){
			for (int j=0;j<size;j++){
				if(i==j) continue;
				double postDist = Math.sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j])+(z[i]-z[j])*(z[i]-z[j])+(w[i]-w[j])*(w[i]-w[j]));
				//System.out.println(postDist+" - "+preDistances[i*size+j]);
				error+=Math.abs(postDist-preDistances[i*size+j])/preDistances[i*size+j];
			}
		}
		System.out.println("Error "+error);
	}
	private void copyDtoH(){
		memX.copyDtoH();
		memY.copyDtoH();
		memZ.copyDtoH();
		memW.copyDtoH();
		memVx.copyDtoH();
		memVy.copyDtoH();
		memVz.copyDtoH();
		memVw.copyDtoH();
	}
	public void release(){
		memX.release();
		memY.release();
		memZ.release();
		memW.release();
		memVx.release();
		memVy.release();
		memVz.release();
		memVw.release();
		memPredistances.release();
		stepV.release();
		program.release();
	}

}
