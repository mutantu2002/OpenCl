package home.mutant.opencl.multilayer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageFloat;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryFloat;
import home.mutant.opencl.model.Program;

public class ObtainFilters {
	List<Image> images;
	int dimFilter;
	int imageSize;
	List<Image> clusterImages = new ArrayList<>();
	int noIterations;
	int noClusters;
	int batchItems = 256*30;
	
	float[] inputImages;
	float[] clustersCenters;
	float[] clustersUpdates;
	
	Program program;
	
	MemoryFloat memClusters;
	MemoryFloat memImages;
	MemoryFloat memUpdates;
	Kernel updateCenters;
	
	public ObtainFilters(List<Image> images, int dimFilter, int noClusters, int noIterations) {
		super();
		this.images = images;
		this.dimFilter = dimFilter;
		this.noIterations = noIterations;
		this.noClusters = noClusters;
		this.imageSize = images.get(0).getDataFloat().length;
		inputImages= new float[imageSize*batchItems];
		
		clustersCenters = new float[dimFilter*dimFilter*noClusters];
		randomizeClusters();
		clustersUpdates = new float[(dimFilter*dimFilter+1)*noClusters*batchItems];
	}
	public void cluster(){
		prepareOpenCl();
		for (int iteration=0;iteration<noIterations;iteration++){
			Arrays.fill(clustersUpdates, 0);
			memUpdates.copyHtoD();
			for (int batch=0 ;batch<images.size()/batchItems;batch++){
				for (int i=0;i<batchItems;i++){
					System.arraycopy(MnistDatabase.trainImages.get(batch*batchItems+i).getDataFloat(), 0, inputImages, i*imageSize, imageSize);
				}
				memImages.copyHtoD();
				updateCenters.run(batchItems, 256);
				program.finish();
			}
			System.out.println(iteration);
			memUpdates.copyDtoH();
			reduceCenters();
			memClusters.copyHtoD();
		}
		releaseOpenCl();
		contructImageClusters();
	}
	public List<Image> getClusterImages() {
		return clusterImages;
	}
	private void reduceCenters() {
		Arrays.fill(clustersCenters, 0);
		for (int i=0;i<noClusters;i++){
			int noUpdates=0;
			int clusterOffset = dimFilter*dimFilter*i;
			for (int batch=0;batch<batchItems;batch++){
				int batchClusterOffset = (dimFilter*dimFilter+1)*(batch*noClusters+i);
				for(int j=0;j<dimFilter*dimFilter;j++){
					clustersCenters[clusterOffset+j]+=clustersUpdates[batchClusterOffset+j];
				}
				noUpdates+=clustersUpdates[batchClusterOffset+dimFilter*dimFilter];
			}
			if (noUpdates>0){
				for(int j=0;j<dimFilter*dimFilter;j++){
					clustersCenters[clusterOffset+j]/=noUpdates;
				}
			}
		}
	}
	private void randomizeClusters() {
		for (int i = 0; i < clustersCenters.length; i++) {
			clustersCenters[i] = (float) (Math.random()*256);
		}
	}
	
	private void prepareOpenCl(){
		Map<String, Object> params = new HashMap<>();
		params.put("IMAGE_SIZE", imageSize);
		params.put("FILTER_SIZE", dimFilter*dimFilter);
		params.put("NO_CLUSTERS", noClusters);
		params.put("DIM_FILTER", dimFilter);
		params.put("DIM_IMAGE", (int)Math.sqrt(imageSize));
		params.put("STRIDE", 1);
		program = new Program(Program.readResource("/opencl/SubImageKmeans2.c"),params);
		
		memClusters = new MemoryFloat(program);
		memClusters.addReadWrite(clustersCenters);
		
		memImages = new MemoryFloat(program);
		memImages.addReadOnly(inputImages);
		
		memUpdates = new MemoryFloat(program);
		memUpdates.addReadWrite(clustersUpdates);
		
		updateCenters = new Kernel(program, "updateCenters");
		updateCenters.setArguments(memClusters,memImages,memUpdates);
	}
	public void releaseOpenCl(){
		memClusters.release();
		memImages.release();
		memUpdates.release();
		updateCenters.release();
		program.release();
	}
	private void contructImageClusters(){
		for (int i=0;i<noClusters;i++) {
			Image image = new ImageFloat(dimFilter*dimFilter);
			System.arraycopy(memClusters.getSrc(), i*dimFilter*dimFilter, image.getDataFloat(), 0, dimFilter*dimFilter);
			clusterImages.add(image);
		}
	}
}
