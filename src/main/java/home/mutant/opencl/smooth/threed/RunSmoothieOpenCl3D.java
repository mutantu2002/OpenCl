package home.mutant.opencl.smooth.threed;

import home.mutant.dl.utils.Utils;
import home.mutant.dl.utils.kmeans.model.ListClusterable;

public class RunSmoothieOpenCl3D {
	public static final int FRAMES = 3000;
	public static void main(String[] args) {
		
		ListClusterable filters = ListClusterable.load("clusters4_256");
		LinkedClusterablesOpenCl3D sm = new LinkedClusterablesOpenCl3D(filters);
		long t0 = System.currentTimeMillis();
		for(int i=0;i<FRAMES;i++){
			sm.stepV();
			sm.show();
			if(i%100==0)System.out.println(i);
		}
		System.out.println("FPS:" + (1000.*FRAMES/(System.currentTimeMillis()-t0)));
		sm.listDistances();
		sm.release();
		Utils.save("smoothclusters4_256_3D", sm);
	}
}
