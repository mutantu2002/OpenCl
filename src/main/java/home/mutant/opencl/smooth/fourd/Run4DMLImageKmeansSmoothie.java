package home.mutant.opencl.smooth.fourd;

import java.util.ArrayList;
import java.util.List;

import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.Utils;
import home.mutant.dl.utils.kmeans.mains.RunImageKmeans;
import home.mutant.dl.utils.kmeans.model.Clusterable;
import home.mutant.dl.utils.kmeans.model.ListClusterable;
import home.mutant.dl.utils.kmeans.model.SimpleClusterable;
import home.mutant.dl.utils.multithreading.Launcher;


public class Run4DMLImageKmeansSmoothie {
	private static final int NO_THREADS = 8;
	private static final int STRIDE = 4;
	
	public static void main(String[] args) throws Exception {
		MnistDatabase.loadImages();
		List<Clusterable> clusterables = new ArrayList<Clusterable>();
		for (int i = 0; i < 60000; i++) {
			clusterables.add(new SimpleClusterable(MnistDatabase.trainImages.get(i).getDataDouble(),MnistDatabase.trainLabels.get(i)));
		}
		LinkedClusterablesOpenCl4D filters = (LinkedClusterablesOpenCl4D) Utils.load("smoothclusters4_256_4D");
		System.out.println(filters.filters.clusterables.size());
		System.out.println(filters.filters.clusterables.get(0).getWeights().length);
		filters.showFilters();
		Launcher launcher = new Launcher();
		int step = clusterables.size() / NO_THREADS;
		
		for (int i = 0; i < NO_THREADS; i++) {
			launcher.addRunnable(new Transform4DClusterablesRunnable(clusterables.subList(i*step, (i+1)*step), filters, STRIDE));
		}
		launcher.run();
		ListClusterable results = new ListClusterable();
		results.clusterables = clusterables.subList(0, 100);
		results.show();
		
		List<Clusterable> clusterablesTest = new ArrayList<Clusterable>();
		for (int i = 0; i < 10000; i++) {
			clusterablesTest.add(new SimpleClusterable(MnistDatabase.testImages.get(i).getDataDouble()));
		}
		
		launcher = new Launcher();
		step = clusterablesTest.size() / NO_THREADS;
		
		for (int i = 0; i < NO_THREADS; i++) {
			launcher.addRunnable(new Transform4DClusterablesRunnable(clusterablesTest.subList(i*step, (i+1)*step), filters, STRIDE));
		}
		launcher.run();
		System.out.println("Start kmeans");
		System.out.println(RunImageKmeans.run(clusterables,clusterablesTest));
	}
}
