package home.mutant.dl.smooth;

import java.io.Serializable;

import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.kmeans.model.Clusterable;
import home.mutant.dl.utils.kmeans.model.ListClusterable;

public class LinkedClusterables implements Serializable{
	private static final long serialVersionUID = 3560289806939024957L;
	public transient double preDistances[][];
	ListClusterable filters;
	double[] x;
	double[] y;
	transient double[] vx;
	transient double[] vy;
	transient double dt=0.0002;
	transient double K=1;
	transient double friction=0.3;
	
	transient ResultFrame frame;
	
	public LinkedClusterables(ListClusterable clusterables) {
		super();
		this.filters = clusterables;
		fillPreDistances();
		x = new double[filters.clusterables.size()];
		y = new double[filters.clusterables.size()];
		vx = new double[filters.clusterables.size()];
		vy = new double[filters.clusterables.size()];
		frame = new ResultFrame(800, 800);
		randDistances();
	}
	private  void fillPreDistances(){
		preDistances = new double[filters.clusterables.size()][filters.clusterables.size()];
		for (int i=0;i<filters.clusterables.size();i++){
			for (int j=0;j<filters.clusterables.size();j++){
				preDistances[i][j]=filters.clusterables.get(i).d(filters.clusterables.get(j))/5;
			}
		}
	}
	private void randDistances(){
		for (int i=0;i<filters.clusterables.size();i++){
			x[i]=Math.random()*200-100;
			y[i]=Math.random()*200-100;
		}
	}
	public void show(){
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
	
	public void stepX(){
		for (int i=0;i<filters.clusterables.size();i++){
			x[i]+=vx[i]*dt;
			y[i]+=vy[i]*dt;
		}
	}
	
	public void stepV(){
		for (int i=0;i<filters.clusterables.size();i++){
			double fx=0;
			double fy=0;
			for (int j=0;j<filters.clusterables.size();j++){
				if (i==j)continue;
				double d = Math.sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]));
				fx+=(x[i]-x[j])/d*(preDistances[i][j]-d)*K;
				fy+=(y[i]-y[j])/d*(preDistances[i][j]-d)*K;
			}
			fx-=vx[i]*friction;
			fy-=vy[i]*friction;
			vx[i]+=fx*dt;
			vy[i]+=fy*dt;
		}
	}
	
	public void listDistances(){
		for (int i=0;i<filters.clusterables.size();i++){
			for (int j=0;j<filters.clusterables.size();j++){
				System.out.println(Math.sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))+" - "+preDistances[i][j]);
			}
		}
	}
	
}
