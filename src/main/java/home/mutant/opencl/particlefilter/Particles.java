package home.mutant.opencl.particlefilter;

public class Particles {
	int[] x;
	int[] y;
	int[] xTmp;
	int[] yTmp;
	double[] weights;
	public Particles(int noParticles) {
		x= new int[noParticles];
		y= new int[noParticles];
		xTmp= new int[noParticles];
		yTmp= new int[noParticles];
		weights= new double[noParticles];
	}
	public void copyToTmp() {
		System.arraycopy(x, 0, xTmp, 0, x.length);
		System.arraycopy(y, 0, yTmp, 0, y.length);
	}
}
