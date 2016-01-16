package home.mutant.opencl.smooth.threed;

import java.util.List;

import home.mutant.dl.models.Image;
import home.mutant.dl.utils.kmeans.model.Clusterable;
import home.mutant.dl.utils.kmeans.model.SimpleClusterable;

public class Clusterable3DSmoothie extends SimpleClusterable {
	private static final long serialVersionUID = -996388060421011310L;
	public double[][] weights3d;
	public Clusterable3DSmoothie(double[] weights) {
		super(weights);
	}
	public Clusterable3DSmoothie(int size) {
		super(size);
		weights3d = new double [3][size];
	}
	public Clusterable3DSmoothie(double[] newImage, int label) {
		super(newImage, label);
	}

	public Clusterable3DSmoothie(double[][] newImage, int label) {
		super(newImage[0],label);
		weights3d=newImage;
	}
	@Override
	public double d(Clusterable clusterable) {
		Clusterable3DSmoothie clusterable3d=(Clusterable3DSmoothie) clusterable;
		
		double d=0;
		for (int i = 0; i < weights.length; i++) {
			d+=(weights3d[0][i]-clusterable3d.weights3d[0][i])*(weights3d[0][i]-clusterable3d.weights3d[0][i])+
					(weights3d[1][i]-clusterable3d.weights3d[1][i])*(weights3d[1][i]-clusterable3d.weights3d[1][i])+
					(weights3d[2][i]-clusterable3d.weights3d[2][i])*(weights3d[2][i]-clusterable3d.weights3d[2][i]);
		}
		return Math.sqrt(d);
	}
	@Override
	public Clusterable copy() {
		Clusterable3DSmoothie m = new Clusterable3DSmoothie(weights.length);
		System.arraycopy(weights3d[0], 0, m.weights3d[0], 0, weights.length);
		System.arraycopy(weights3d[1], 0, m.weights3d[1], 0, weights.length);
		System.arraycopy(weights3d[2], 0, m.weights3d[2], 0, weights.length);
		return m;
	}
	@Override
	public void updateCenterFromMembers(List<Clusterable> allList, List<Integer> cluster) {
		for(int w=0; w<weights.length; w++)
		{
			double w0=0;
			double w1=0;
			double w2=0;
			for (int i = 0; i<cluster.size(); i++)
			{
				w0+=((Clusterable3DSmoothie)allList.get(cluster.get(i))).weights3d[0][w];
				w1+=((Clusterable3DSmoothie)allList.get(cluster.get(i))).weights3d[1][w];
				w2+=((Clusterable3DSmoothie)allList.get(cluster.get(i))).weights3d[2][w];
			}
			weights3d[0][w]= (w0/cluster.size());
			weights3d[1][w]= (w1/cluster.size());
			weights3d[2][w]= (w2/cluster.size());
		}
		System.arraycopy(weights3d[0], 0, weights, 0, weights.length);
	}
	@Override
	public Image getImage(){
		return super.getImage();
	}
}
