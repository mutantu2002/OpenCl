package home.mutant.dl.smooth;

import home.mutant.dl.utils.kmeans.model.ListClusterable;
import home.mutant.dl.utils.kmeans.smooth.LinkedClusterables;

public class RunSmoothie {
	public static final int FRAMES = 20000;
	public static void main(String[] args) {
		
		ListClusterable filters = ListClusterable.load("clusters4_256");
		filters.clusterables = filters.clusterables.subList(0, 256);
		LinkedClusterables sm = new LinkedClusterables(filters);
		long t0 = System.currentTimeMillis();
		for(int i=0;i<FRAMES;i++){
			sm.stepV();
			sm.stepX();
			if(i%1000==0)
				sm.show();
			if(i%1000==0)System.out.println(i);
		}
		System.out.println("FPS:" + (1000.*FRAMES/(System.currentTimeMillis()-t0)));
		//sm.listDistances();
	}

}
