package home.mutant.opencl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageFloat;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.dl.utils.kmeans.Kmeans;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryFloat;
import home.mutant.opencl.model.MemoryInt;
import home.mutant.opencl.model.Program;

public class KMeansOpenCl2 {
	public static final int NO_CLUSTERS = 256;
	public static final int WORK_ITEMS = 256*234;
	public static final int NO_ITERATIONS = 50;
	public static final int IMAGE_SIZE = 784;
	
	public static void main(String[] args) throws Exception {
		
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImages();
		
		float[] images= new float[IMAGE_SIZE*WORK_ITEMS];
		
		float[] clustersCenters = new float[IMAGE_SIZE*NO_CLUSTERS];
		randomizeCentersFromImages(clustersCenters);
		int[] clustersUpdates = new int[WORK_ITEMS];
		
		Map<String, Object> params = new HashMap<>();
		params.put("imageSize", IMAGE_SIZE);
		Program program = new Program(Program.readResource("/opencl/Kmeans2.c"),params);		
		
		MemoryFloat memClusters = new MemoryFloat(program);
		memClusters.addReadWrite(clustersCenters);
		
		MemoryFloat memImages = new MemoryFloat(program);
		memImages.addReadOnly(images);
		
		MemoryInt memUpdates = new MemoryInt(program);
		memUpdates.addReadWrite(clustersUpdates);

		Kernel updateCenters = new Kernel(program, "updateCenters");
		updateCenters.setArgument(memClusters,0);
		updateCenters.setArgument(memImages,1);
		updateCenters.setArgument(memUpdates,2);
		updateCenters.setArgument(NO_CLUSTERS, 3);
		
		long tTotal=0;
		for (int i=0;i<WORK_ITEMS;i++){
			System.arraycopy(MnistDatabase.trainImages.get(i).getDataFloat(), 0, images, i*(IMAGE_SIZE), IMAGE_SIZE);
		}
		memImages.copyHtoD();
		
		for (int iteration=0;iteration<NO_ITERATIONS;iteration++){
			long t0 = System.currentTimeMillis();
			
			updateCenters.run(WORK_ITEMS, 256);
			program.finish();
			memUpdates.copyDtoH();
			reduceCenters(images, clustersCenters, clustersUpdates);
			memClusters.copyHtoD();
			tTotal+=System.currentTimeMillis()-t0;
			
			System.out.println("Iteration "+iteration);
		}
		System.out.println("Time in kernel per iteration " + tTotal/1000./NO_ITERATIONS);
		
		memUpdates.copyDtoH();
		memClusters.copyDtoH();
		
		memClusters.release();
		memImages.release();
		memUpdates.release();
		updateCenters.release();
		program.release();
		
		List<HashMap<Integer, Integer>> clusterHash = new ArrayList<>();
		for (int i=0;i<NO_CLUSTERS;i++) {
			clusterHash.add(new HashMap<>());
		}
		for (int i=0;i<WORK_ITEMS;i++){
			int currentClusterIndex = clustersUpdates[i];
			int currentLabel = MnistDatabase.trainLabels.get(i);
			HashMap<Integer, Integer> currentMembers =  clusterHash.get(currentClusterIndex);
			if (currentMembers.get(currentLabel)==null){
				currentMembers.put(currentLabel, 0);
			}
			currentMembers.put(currentLabel, currentMembers.get(currentLabel)+1);
		}
		List<Integer> clusterLabels = new ArrayList<>();
		for (int i=0;i<NO_CLUSTERS;i++) {
			clusterLabels.add(Kmeans.getMaxKeyHash(clusterHash.get(i)));
		}
		System.out.println(Arrays.toString(clusterLabels.toArray()));
		List<Image> imagesClusters = new ArrayList<Image>();
		for (int i=0;i<NO_CLUSTERS;i++) {
			Image image = new ImageFloat(IMAGE_SIZE);
			System.arraycopy(memClusters.getSrc(), i*IMAGE_SIZE, image.getDataFloat(), 0, IMAGE_SIZE);
			imagesClusters.add(image);
		}
		ResultFrame frame = new ResultFrame(600, 600);
		frame.showImages(imagesClusters);

	}

	private static void reduceCenters(float[] images, float[] clustersCenters, int[] clustersUpdates) {
		int[] clustersMembers = new int[NO_CLUSTERS];
		for (int i=0;i<WORK_ITEMS;i++){
			int toUpdate = clustersUpdates[i];
			clustersMembers[toUpdate]++;
			for (int j=0;j<IMAGE_SIZE;j++){
				clustersCenters[toUpdate*IMAGE_SIZE+j]+=images[i*IMAGE_SIZE+j];
			}
		}
		for (int i=0;i<NO_CLUSTERS;i++){
			if (clustersMembers[i]==0) continue;
			for (int j=0;j<IMAGE_SIZE;j++){
				clustersCenters[i*IMAGE_SIZE+j]/=clustersMembers[i];
			}
		}
	}

	private static void randomizeCenters(float[] clustersCenters) {
		for (int i = 0; i < clustersCenters.length; i++) {
			clustersCenters[i] = (float) (Math.random()*256);
		}
	}
	private static void randomizeCentersFromImages(float[] clustersCenters) {
		for (int i=0;i<NO_CLUSTERS;i++){
			System.arraycopy(MnistDatabase.trainImages.get((int) (Math.random()*WORK_ITEMS)).getDataFloat(), 0, clustersCenters, i*(IMAGE_SIZE), IMAGE_SIZE);
		}
	}
}
