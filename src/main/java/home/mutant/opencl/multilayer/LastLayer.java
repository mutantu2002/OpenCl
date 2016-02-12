package home.mutant.opencl.multilayer;

import java.util.List;

import home.mutant.dl.models.Image;
import home.mutant.opencl.multilayer.steps.ClusterImages;

public class LastLayer {
	List<Image> trainImages;
	List<Image> testImages;
	List<Integer> trainLabels;
	List<Integer> testLabels;
	int noClusters;
	int noIterations;
	
	public LastLayer(List<Image> trainImages, List<Image> testImages, List<Integer> trainLabels,
			List<Integer> testLabels, int noClusters, int noIterations) {
		super();
		this.trainImages = trainImages;
		this.testImages = testImages;
		this.trainLabels = trainLabels;
		this.testLabels = testLabels;
		this.noClusters = noClusters;
		this.noIterations = noIterations;
	}
	public List<Image> test(){
		ClusterImages  ci = new ClusterImages(trainImages, trainLabels, noClusters, noIterations);
		ci.cluster();
		ci.test(testImages, testLabels);
		ci.releaseOpenCl();
		return ci.getClusters();
	}
}
