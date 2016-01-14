package home.mutant.dl.smooth;

import home.mutant.dl.utils.Utils;
import home.mutant.dl.utils.kmeans.model.ListClusterable;
import home.mutant.dl.smooth.LinkedClusterablesOpenCl;

public class RunSmoothieOpenCl {
	public static final int FRAMES = 3000;
	public static void main(String[] args) {
		
		ListClusterable filters = ListClusterable.load("clusters4_256");
		LinkedClusterablesOpenCl sm = new LinkedClusterablesOpenCl(filters);
		long t0 = System.currentTimeMillis();
		for(int i=0;i<FRAMES;i++){
			sm.stepV();
			//if(i%1==0)
			sm.show();
			if(i%100==0)System.out.println(i);
		}
		System.out.println("FPS:" + (1000.*FRAMES/(System.currentTimeMillis()-t0)));
		//sm.listDistances();
		sm.release();
		Utils.save("smoothclusters4_256", sm);
	}
}
