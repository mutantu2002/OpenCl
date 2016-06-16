package home.mutant.opencl.dot.steps;

import java.util.List;

import home.mutant.dl.models.Image;

public class MaxIndexImage {

	public int[] maxIndexes;
	public double rate;
	public MaxIndexImage(List<Image> images, List<Integer> labels, int clustersPerType) {
		int imageSize = images.get(0).getDataShort().length;
		maxIndexes = new int[images.size()];
		for (int i = 0; i < maxIndexes.length; i++) {
			int max=Integer.MIN_VALUE;
			int indexMax=0;
			for (int j = 0; j < imageSize; j++) {
				if(images.get(i).getDataShort()[j]>max){
					max=(int) images.get(i).getDataShort()[j];
					indexMax=j;
				}
			}
			maxIndexes[i]=indexMax/clustersPerType;
		}
		int count=0;
		for (int i = 0; i < maxIndexes.length; i++) {
			if(maxIndexes[i]==labels.get(i)) count++;
		}
		rate = count*100./labels.size();
	}
	
}
