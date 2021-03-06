package home.mutant.opencl.smooth.threed;

import java.util.List;

import home.mutant.dl.utils.ImageUtils;
import home.mutant.dl.utils.kmeans.model.Clusterable;
import home.mutant.dl.utils.kmeans.model.SimpleClusterable;

public class Transform3DClusterablesRunnable implements Runnable{
	List<Clusterable> toTransform;
	LinkedClusterablesOpenCl3D clusters;
	int stride;
	
	public Transform3DClusterablesRunnable(List<Clusterable> toTransform, LinkedClusterablesOpenCl3D clusters, int stride) {
		super();
		this.toTransform = toTransform;
		this.clusters = clusters;
		this.stride = stride;
	}

	public Transform3DClusterablesRunnable(List<Clusterable> toTransform, LinkedClusterablesOpenCl3D clusters) {
		this(toTransform, clusters,1);
	}
	
	@Override
	public void run() {
		int sizeSubImage = (int) Math.sqrt(clusters.filters.clusterables.get(0).getWeights().length);
		int imageSize = (int) Math.sqrt(toTransform.get(0).getWeights().length);
		int newImageSize = (imageSize - sizeSubImage)/stride+1;
		for(int i = 0;i<toTransform.size();i++){
			Clusterable current = toTransform.get(i);
			List<double[]> dividedImages = ImageUtils.divideImage(current.getWeights(), sizeSubImage, sizeSubImage, 
					imageSize, imageSize, stride, stride);
			double[][] newImage = new double[3][newImageSize*newImageSize];
			for (int j = 0; j < newImage[0].length; j++) {
				SimpleClusterable sc = new SimpleClusterable(dividedImages.get(j));
				int indexCluster = clusters.filters.getClosestClusterIndex( sc);
				newImage[0][j]=clusters.x[indexCluster];
				newImage[1][j]=clusters.y[indexCluster];
				newImage[2][j]=clusters.z[indexCluster];
			}
			toTransform.set(i, new Clusterable3DSmoothie(newImage,current.getLabel()));
		}
		
	}

}
