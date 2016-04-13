package home.mutant.opencl.perceptron.runners;

import java.io.IOException;

import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.perceptron.PerceptronOpenCl;

public class RunPerceptronOpenCl {
	public static void main(String[] args) throws IOException {
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImages();
		PerceptronOpenCl pocl = new PerceptronOpenCl(MnistDatabase.trainImages, MnistDatabase.trainLabels);
		pocl.output();
		pocl.calculateEntropy();
		pocl.releaseOpenCl();
	}
}
