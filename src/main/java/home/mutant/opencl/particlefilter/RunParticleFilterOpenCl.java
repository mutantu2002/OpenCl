package home.mutant.opencl.particlefilter;

import java.io.IOException;

import home.mutant.dl.models.Image;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;

public class RunParticleFilterOpenCl {
	private static final int NO_STEPS=6000;
	public static void main(String[] args) throws IOException {
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImagesCrop(20);
		Map map = new Map(MnistDatabase.trainImages, 20);
		Image tofind = MnistDatabase.trainImages.get(1011);
		ParticleFilter pf = new ParticleFilter(map.map, tofind, 200000);
		ResultFrame frame2 = new ResultFrame(1000, 900);
		frame2.showImage(map.map);
		frame2.showImage(tofind, 900, 0);
		ResultFrame frame = new ResultFrame(1000, 900);
		frame.showImage(pf.getImageParticles());
		long t0=System.currentTimeMillis();
		for(int step=0;step<NO_STEPS;step++){
			pf.step();
			//if (step%30==9)
				frame.showImage(pf.getImageParticles());
			//System.out.println(step+":"+pf.noParticles);
		}
		pf.release();
		t0=System.currentTimeMillis()-t0;
		System.out.println("FPS:"+1000.*NO_STEPS/t0);
		
	}

}
