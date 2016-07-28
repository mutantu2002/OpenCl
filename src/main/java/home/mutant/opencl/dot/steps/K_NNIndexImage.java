package home.mutant.opencl.dot.steps;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import home.mutant.dl.models.Image;
import home.mutant.dl.utils.MathUtils;

public class K_NNIndexImage {

	public double[] rate;
	public K_NNIndexImage(List<Image> images, List<Integer> labels, int clustersPerType, int k) {
		int imageSize = images.get(0).getDataShort().length;
		int[] count=new int[k];
		rate=new double[k];
		for (int i = 0; i < images.size(); i++) {
			List<ValueLabel> vl =new ArrayList<>();
			for (int j = 0; j < imageSize; j++) {
				int label = j/clustersPerType;
				vl.add(new ValueLabel(label, (double) images.get(i).getDataShort()[j]));
			}
			Collections.sort(vl,Collections.reverseOrder());
			for(int kk=0;kk<k;kk++){
				if(MathUtils.getKeyForMaxValue(getMapForCounts(vl,kk+1))==labels.get(i)) count[kk]++;
			}
			
		}
		for(int kk=0;kk<k;kk++){
			rate[kk] = count[kk]*100./labels.size();
		}
	}
	public Map<Integer, Integer> getMapForCounts(List<ValueLabel> vls, int kk){
		Map<Integer, Integer> maps= new HashMap<>();
		for(int i=0;i<kk;i++){
			if(maps.get(vls.get(i).label)==null){
				maps.put(vls.get(i).label, 0);
			}
			maps.put(vls.get(i).label, maps.get(vls.get(i).label)+100-i);
		}
		return maps;
	}
	class ValueLabel implements Comparable<ValueLabel>{
		int label;
		Double value;
		public ValueLabel(int label, Double value) {
			super();
			this.label = label;
			this.value = value;
		}
		@Override
		public int compareTo(ValueLabel o) {
			return value.compareTo(o.value);
		}
	}
	
}
