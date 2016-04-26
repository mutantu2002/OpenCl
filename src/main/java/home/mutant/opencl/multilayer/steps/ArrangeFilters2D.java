package home.mutant.opencl.multilayer.steps;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import home.mutant.dl.models.Image;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryFloat;
import home.mutant.opencl.model.Program;

public class ArrangeFilters2D {
	public List<Image> images;
	
	public  float preDistances[];
	public float[] x;
	public float[] y;
	float[] vx;
	float[] vy;
	float dt=0.001f;
	float K=1;
	float friction=0.1f;
	
	int scaleDistances=4;
	ResultFrame frame;
	private  MemoryFloat memX;
	private  MemoryFloat memY;
	private  MemoryFloat memVx;
	private  MemoryFloat memVy;
	private  MemoryFloat memPredistances;
	private  Kernel stepV;
	private  Program program;
	
	public ArrangeFilters2D(List<Image> images) {
		this(images,4);
	}
	public ArrangeFilters2D(List<Image> images, int scaleDistances) {
		super();
		this.images = images;
		this.scaleDistances = scaleDistances;
		fillPreDistances();
		x = new float[images.size()];
		y = new float[images.size()];
		vx = new float[images.size()];
		vy = new float[images.size()];
		frame = new ResultFrame(800, 800);
		randDistances();
		
		Map<String, Object> params = new HashMap<>();
		params.put("NO_PARTICLES", images.size());
		params.put("DT", dt);
		params.put("K", K);
		params.put("FRICTION", friction);
		params.put("BATCH", images.size()/256);
		program = new Program(Program.readResource("/opencl/ElasticSmoothie2DFloat.c"),params);
		
		memX = new MemoryFloat(program);
		memX.addReadWrite(x);
		
		memY = new MemoryFloat(program);
		memY.addReadWrite(y);
		
		memVx = new MemoryFloat(program);
		memVx.addReadWrite(vx);
		
		memVy = new MemoryFloat(program);
		memVy.addReadWrite(vy);
		
		memPredistances = new MemoryFloat(program);
		memPredistances.addReadOnly(preDistances);
		
		stepV = new Kernel(program, "stepV");
		stepV.setArguments(memX,memY,memVx, memVy, memPredistances);
		
	}
	private  void fillPreDistances(){
		int size = images.size();
		preDistances = new float[size*size];
		for (int i=0;i<size;i++){
			for (int j=0;j<size;j++){
				preDistances[i*size+j]=(float) (d(images.get(i),images.get(j))/scaleDistances);
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
			x[i]=(float) (Math.random()*300-150);
			y[i]=(float) (Math.random()*300-150);
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

	public double getMediumV(){
		double v=0;
		for (int i=0;i<images.size();i++){
			v+=Math.sqrt(vx[i]*vx[i]+vy[i]*vy[i]);
		}
		return v/images.size();
	}
	public void stepV(){
		stepV.run(256, 256);
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
				double postDist = Math.sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]));
				//System.out.println(postDist+" - "+preDistances[i*size+j]);
				error+=Math.abs(postDist-preDistances[i*size+j])/preDistances[i*size+j];
			}
		}
		System.out.println("Error "+error);
	}
	public void copyDtoH(){
		memX.copyDtoH();
		memY.copyDtoH();
		memVx.copyDtoH();
		memVy.copyDtoH();
	}
	public void release(){
		memX.release();
		memY.release();
		memVx.release();
		memVy.release();
		memPredistances.release();
		stepV.release();
		program.release();
	}
}
