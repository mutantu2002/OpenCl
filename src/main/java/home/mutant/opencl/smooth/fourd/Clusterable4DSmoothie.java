package home.mutant.opencl.smooth.fourd;

import java.util.List;

import home.mutant.dl.models.Image;
import home.mutant.dl.utils.kmeans.model.Clusterable;
import home.mutant.dl.utils.kmeans.model.SimpleClusterable;

public class Clusterable4DSmoothie extends SimpleClusterable {
	private static final long serialVersionUID = -996388060421011310L;
	public double[][] weights4d;
	public Clusterable4DSmoothie(double[] weights) {
		super(weights);
	}
	public Clusterable4DSmoothie(int size) {
		super(size);
		weights4d = new double [4][size];
	}
	public Clusterable4DSmoothie(double[] newImage, int label) {
		super(newImage, label);
	}

	public Clusterable4DSmoothie(double[][] newImage, int label) {
		super(newImage[0],label);
		weights4d=newImage;
	}
	@Override
	public double d(Clusterable clusterable) {
		Clusterable4DSmoothie clusterable3d=(Clusterable4DSmoothie) clusterable;
		
		double d=0;
		for (int i = 0; i < weights.length; i++) {
			d+=(weights4d[0][i]-clusterable3d.weights4d[0][i])*(weights4d[0][i]-clusterable3d.weights4d[0][i])+
					(weights4d[1][i]-clusterable3d.weights4d[1][i])*(weights4d[1][i]-clusterable3d.weights4d[1][i])+
					(weights4d[2][i]-clusterable3d.weights4d[2][i])*(weights4d[2][i]-clusterable3d.weights4d[2][i])+
					(weights4d[3][i]-clusterable3d.weights4d[3][i])*(weights4d[3][i]-clusterable3d.weights4d[3][i]);
		}
		return Math.sqrt(d);
	}
	@Override
	public Clusterable copy() {
		Clusterable4DSmoothie m = new Clusterable4DSmoothie(weights.length);
		System.arraycopy(weights4d[0], 0, m.weights4d[0], 0, weights.length);
		System.arraycopy(weights4d[1], 0, m.weights4d[1], 0, weights.length);
		System.arraycopy(weights4d[2], 0, m.weights4d[2], 0, weights.length);
		System.arraycopy(weights4d[3], 0, m.weights4d[3], 0, weights.length);
		return m;
	}
	@Override
	public void updateCenterFromMembers(List<Clusterable> allList, List<Integer> cluster) {
		for(int w=0; w<weights.length; w++)
		{
			double w0=0;
			double w1=0;
			double w2=0;
			double w3=0;
			for (int i = 0; i<cluster.size(); i++)
			{
				w0+=((Clusterable4DSmoothie)allList.get(cluster.get(i))).weights4d[0][w];
				w1+=((Clusterable4DSmoothie)allList.get(cluster.get(i))).weights4d[1][w];
				w2+=((Clusterable4DSmoothie)allList.get(cluster.get(i))).weights4d[2][w];
				w3+=((Clusterable4DSmoothie)allList.get(cluster.get(i))).weights4d[3][w];
			}
			weights4d[0][w]= (w0/cluster.size());
			weights4d[1][w]= (w1/cluster.size());
			weights4d[2][w]= (w2/cluster.size());
			weights4d[3][w]= (w3/cluster.size());
		}
		System.arraycopy(weights4d[0], 0, weights, 0, weights.length);
	}
	@Override
	public Image getImage(){
		return super.getImage();
	}
}
