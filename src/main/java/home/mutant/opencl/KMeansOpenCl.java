package home.mutant.opencl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageDouble;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.ImageUtils;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryDouble;
import home.mutant.opencl.model.Program;

public class KMeansOpenCl {
	public static final int DIM_FILTER = 4;
	public static final int NO_CLUSTERS = 256;
	public static final int IMAGES_PER_WORK_ITEM = 625;
	public static final int WORK_ITEMS = 2560;
	public static final int NO_ITERATIONS = 5;
	
	public static void main(String[] args) throws Exception {
		MnistDatabase.loadImages();
		double[] subImages= new double[(DIM_FILTER*DIM_FILTER)*IMAGES_PER_WORK_ITEM*WORK_ITEMS];
		
		double[] clustersCenters = new double[DIM_FILTER*DIM_FILTER*NO_CLUSTERS];
		for (int i = 0; i < clustersCenters.length; i++) {
			clustersCenters[i] = Math.random()*256;
		}
		double[] clustersUpdates = new double[(DIM_FILTER*DIM_FILTER+1)*NO_CLUSTERS*WORK_ITEMS];
		
		Program program = new Program(Program.readResource("/opencl/Kmeans.c"));
		
		MemoryDouble memClusters = new MemoryDouble(program);
		memClusters.addReadWrite(clustersCenters);
		
		MemoryDouble memImages = new MemoryDouble(program);
		memImages.addReadOnly(subImages);
		
		MemoryDouble memUpdates = new MemoryDouble(program);
		memUpdates.addReadWrite(clustersUpdates);
		
		Kernel updateCenters = new Kernel(program, "updateCenters");
		updateCenters.setArguments(memClusters,memImages,memUpdates);
		
		Kernel reduceCenters = new Kernel(program, "reduceCenters");
		reduceCenters.setArguments(memClusters,memUpdates);
		long tTotal=0;

		for (int iteration=0;iteration<NO_ITERATIONS;iteration++){
			Arrays.fill(clustersUpdates, 0);
			memUpdates.copyHtoD();
			for (int batch=0 ;batch<60000/WORK_ITEMS;batch++){
				for (int i=0;i<WORK_ITEMS;i++){
					System.arraycopy(ImageUtils.divideSquareImageUnidimensional(MnistDatabase.trainImages.get(batch*WORK_ITEMS+i).getDataDouble(), DIM_FILTER), 0, subImages, i*(DIM_FILTER*DIM_FILTER)*625, (DIM_FILTER*DIM_FILTER)*625);
				}
				long t0 = System.currentTimeMillis();
				memImages.copyHtoD();
				updateCenters.run(WORK_ITEMS, 128);
				program.finish();
				tTotal+=System.currentTimeMillis()-t0;
			}
			reduceCenters.run(NO_CLUSTERS, 128);
			program.finish();

			System.out.println("Iteration "+iteration);
		}
		memUpdates.copyDtoH();
		double sum = 0;
		int noZero=0;
		for (int i =16;i<(DIM_FILTER*DIM_FILTER+1)*NO_CLUSTERS*WORK_ITEMS;i+=17){
			if (memUpdates.getSrc()[i]==0)noZero++;
			sum+=memUpdates.getSrc()[i];
		}
		System.out.println("Threaded Clusters not assigned "+noZero);
		System.out.println("Total updates "+sum);
		System.out.println("Time in kernel per iteration " + tTotal/1000./NO_ITERATIONS);
		memClusters.copyDtoH();
		List<Image> images = new ArrayList<Image>();
		for (int i=0;i<NO_CLUSTERS;i++) {
			Image image = new ImageDouble(DIM_FILTER*DIM_FILTER);
			System.arraycopy(memClusters.getSrc(), i*DIM_FILTER*DIM_FILTER, image.getDataDouble(), 0, DIM_FILTER*DIM_FILTER);
			images.add(image);
		}
		ResultFrame frame = new ResultFrame(600, 600);
		frame.showImages(images);
		memClusters.release();
		memImages.release();
		memUpdates.release();
		updateCenters.release();
		program.release();
	}
}
