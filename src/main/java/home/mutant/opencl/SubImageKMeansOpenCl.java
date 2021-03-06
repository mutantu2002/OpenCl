package home.mutant.opencl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageFloat;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryFloat;
import home.mutant.opencl.model.Program;
import home.mutant.dl.utils.Utils;

public class SubImageKMeansOpenCl {
	public static final int DIM_FILTER = 16;
	public static final int NO_CLUSTERS = 256;
	public static final int WORK_ITEMS = 256*30;
	public static final int NO_ITERATIONS = 2;
	
	public static final int WORK_GROUP_SIZE = 256;
	
	public static final int DIM_IMAGE = 28*5;
	public static final int NO_MNIST_IMAGES = 60000;
	
	public static void main(String[] args) throws Exception {
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImagesPadded(56);
		float[] inputImages= new float[(DIM_IMAGE*DIM_IMAGE)*WORK_ITEMS];
		
		float[] clustersCenters = new float[DIM_FILTER*DIM_FILTER*NO_CLUSTERS];
		for (int i = 0; i < clustersCenters.length; i++) {
			clustersCenters[i] = (float) (Math.random()*256);
		}
		float[] clustersUpdates = new float[(DIM_FILTER*DIM_FILTER+1)*NO_CLUSTERS*WORK_ITEMS];
		
		Program program = new Program(Program.readResource("/opencl/SubImageKmeans.c"));
		
		MemoryFloat memClusters = new MemoryFloat(program);
		memClusters.addReadWrite(clustersCenters);
		
		MemoryFloat memImages = new MemoryFloat(program);
		memImages.addReadOnly(inputImages);
		
		MemoryFloat memUpdates = new MemoryFloat(program);
		memUpdates.addReadWrite(clustersUpdates);
		
		Kernel updateCenters = new Kernel(program, "updateCenters");
		updateCenters.setArguments(memClusters,memImages,memUpdates);
		//updateCenters.set2Argument(DIM_FILTER,NO_CLUSTERS, 3);
		
		Kernel reduceCenters = new Kernel(program, "reduceCenters");
		reduceCenters.setArguments(memUpdates);
		
		int dimNoClusters = (int) Math.sqrt(NO_CLUSTERS);
		Kernel mixCenters = new Kernel(program, "mixCenters2D");
		mixCenters.setArguments(memClusters, memUpdates);
		mixCenters.setArgument(dimNoClusters,2);
		
		long tTotal=0;

		long t0 = System.currentTimeMillis();
		for (int iteration=0;iteration<NO_ITERATIONS;iteration++){
			Arrays.fill(clustersUpdates, 0);
			memUpdates.copyHtoD();
			for (int batch=0 ;batch<NO_MNIST_IMAGES/WORK_ITEMS;batch++){
				for (int i=0;i<WORK_ITEMS;i++){
					System.arraycopy(MnistDatabase.trainImages.get(batch*WORK_ITEMS+i).getDataFloat(), 0, inputImages, i*(DIM_IMAGE*DIM_IMAGE), DIM_IMAGE*DIM_IMAGE);
				}
				
				memImages.copyHtoD();
				updateCenters.run(WORK_ITEMS, WORK_GROUP_SIZE);
				program.finish();
				
			}
			//memUpdates.copyDtoH();
			//reduceCenters(clustersCenters, clustersUpdates);
			//memClusters.copyHtoD();
			reduceCenters.run(NO_CLUSTERS, NO_CLUSTERS);
			program.finish();
			mixCenters.run(NO_CLUSTERS, NO_CLUSTERS);
			program.finish();
			System.out.println("Iteration "+iteration);
		}
		tTotal+=System.currentTimeMillis()-t0;
		memUpdates.copyDtoH();
		double sum = 0;
		int noZero=0;
		for (int i =16;i<(DIM_FILTER*DIM_FILTER+1)*NO_CLUSTERS*WORK_ITEMS;i+=17){
			if (memUpdates.getSrc()[i]==0)noZero++;
			sum+=memUpdates.getSrc()[i];
		}
		System.out.println("Threaded Clusters not assigned "+noZero);
		System.out.println("Total updates "+sum);
		System.out.println("Time in per iteration " + tTotal/1000./NO_ITERATIONS);
		memClusters.copyDtoH();
		List<Image> imgClusters = new ArrayList<Image>();
		for (int i=0;i<NO_CLUSTERS;i++) {
			Image image = new ImageFloat(DIM_FILTER*DIM_FILTER);
			System.arraycopy(memClusters.getSrc(), i*DIM_FILTER*DIM_FILTER, image.getDataFloat(), 0, DIM_FILTER*DIM_FILTER);
			imgClusters.add(image);
		}
		
		ResultFrame frame = new ResultFrame(600, 600);
		frame.showImages(imgClusters,dimNoClusters);
		
		Utils.save("clusters"+DIM_FILTER+"_"+NO_CLUSTERS, imgClusters.toArray(new Image[0]));
		memClusters.release();
		memImages.release();
		memUpdates.release();
		updateCenters.release();
		reduceCenters.release();
		mixCenters.release();
		program.release();
	}
	private static void reduceCenters(float[] clustersCenters, float[] clustersUpdates) {
		int filterSize=DIM_FILTER*DIM_FILTER;
		Arrays.fill(clustersCenters, 0);
		for (int i=0;i<NO_CLUSTERS;i++){
			int noUpdates=0;
			int clusterOffset = filterSize*i;
			for (int batch=0;batch<WORK_ITEMS;batch++){
				int batchClusterOffset = (filterSize+1)*(batch*NO_CLUSTERS+i);
				for(int j=0;j<filterSize;j++){
					clustersCenters[clusterOffset+j]+=clustersUpdates[batchClusterOffset+j];
				}
				noUpdates+=clustersUpdates[batchClusterOffset+filterSize];
			}
			if (noUpdates>0){
				for(int j=0;j<filterSize;j++){
					clustersCenters[clusterOffset+j]/=noUpdates;
				}
			}
		}
		int dimNoClusters=8;
		float influence=2f;
		for (int i=0;i<NO_CLUSTERS;i++){
			int offsetCenterX=i%dimNoClusters;
			int offsetCenterY=i/dimNoClusters;
			
			int offsetCenterX1=(offsetCenterX+1)%dimNoClusters;
			int offsetCenterY1=(offsetCenterY+1)%dimNoClusters;
			int offsetCenterX_1=(offsetCenterX+dimNoClusters-1)%dimNoClusters;
			int offsetCenterY_1=(offsetCenterY+dimNoClusters-1)%dimNoClusters;
			int clusterOffset = filterSize*i;
			for(int j=0;j<filterSize;j++){
				clustersUpdates[clusterOffset+j]=influence*clustersCenters[clusterOffset+j];
				clustersUpdates[clusterOffset+j]+=clustersCenters[(offsetCenterY*dimNoClusters+offsetCenterX1)*filterSize+j];
				clustersUpdates[clusterOffset+j]+=clustersCenters[(offsetCenterY*dimNoClusters+offsetCenterX_1)*filterSize+j];
				clustersUpdates[clusterOffset+j]+=clustersCenters[(offsetCenterY1*dimNoClusters+offsetCenterX)*filterSize+j];
				clustersUpdates[clusterOffset+j]+=clustersCenters[(offsetCenterY_1*dimNoClusters+offsetCenterX)*filterSize+j];
				clustersUpdates[clusterOffset+j]/=influence+4;
			}
		}
		System.arraycopy(clustersUpdates, 0, clustersCenters, 0,  filterSize*NO_CLUSTERS);
	}
}
