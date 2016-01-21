package home.mutant.opencl.smooth.fourd;

import home.mutant.dl.utils.Utils;
import home.mutant.dl.utils.kmeans.model.ListClusterable;

public class RunSmoothieOpenCl4D {
	public static final int FRAMES = 4000;
	public static void main(String[] args) {
		
		ListClusterable filters = ListClusterable.load("clusters4_256");
		LinkedClusterablesOpenCl4D sm = new LinkedClusterablesOpenCl4D(filters);
		sm.listDistances();
		long t0 = System.currentTimeMillis();
		for(int i=0;i<FRAMES;i++){
			sm.stepV();
			sm.show();
			if(i%100==0)System.out.println(i);
		}
		System.out.println("FPS:" + (1000.*FRAMES/(System.currentTimeMillis()-t0)));
		sm.listDistances();
		sm.release();
		Utils.save("smoothclusters4_256_4D", sm);
	}
}
