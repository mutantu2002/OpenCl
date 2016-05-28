package home.mutant.opencl.dot.steps;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageFloat;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryFloat;
import home.mutant.opencl.model.Program;

public class OnlineUpdateFilters {
	List<Image> images;
	int dimFilter;
	int imageSize;
	List<Image> clusterImages = new ArrayList<>();
	int noIterations=10;
	int noClusters;
	int batchItems = 256;
	int stride=1;
	float[] inputImages;
	float[] clustersCenters;

	Program program;
	
	MemoryFloat memClusters;
	MemoryFloat memImages;
	Kernel updateCenters;
	
	public OnlineUpdateFilters(List<Image> images) {
		super();
		this.images = images;

	}
	public OnlineUpdateFilters build(){
		this.imageSize = images.get(0).getDataFloat().length;
		inputImages= new float[imageSize*batchItems];
		
		clustersCenters = new float[dimFilter*dimFilter*noClusters];
		randomizeClusters();
		subtractMeanClusters();
		return this;
	}
	public OnlineUpdateFilters setNoClusters(int noClusters){
		this.noClusters = noClusters;
		return this;
	}
	public OnlineUpdateFilters setDimFilter(int dimFilter){
		this.dimFilter = dimFilter;
		return this;
	}

	public void cluster(){
		prepareOpenCl();
		for (int batch=0 ;batch<images.size()/batchItems;batch++){
			for (int i=0;i<batchItems;i++){
				System.arraycopy(images.get(batch*batchItems+i).getDataFloat(), 0, inputImages, i*imageSize, imageSize);
			}
			memImages.copyHtoD();
			updateCenters.run(batchItems, noClusters);
			program.finish();
		}

		memClusters.copyDtoH();
		releaseOpenCl();
		constructImageClusters();
	}
	public List<Image> getClusterImages() {
		return clusterImages;
	}

	private void subtractMeanClusters(){
		for (int i=0;i<noClusters;i++){
			int clusterOffset = dimFilter*dimFilter*i;
			double mean=0;
			double lenght=0;
			for(int j=0;j<dimFilter*dimFilter;j++){
				mean+=clustersCenters[clusterOffset+j];
			}
			
			mean/=dimFilter*dimFilter;
			for(int j=0;j<dimFilter*dimFilter;j++){
				clustersCenters[clusterOffset+j]-=mean;
				lenght+=clustersCenters[clusterOffset+j]*clustersCenters[clusterOffset+j];
			}
			lenght = Math.sqrt(lenght);
			for(int j=0;j<dimFilter*dimFilter;j++){
				clustersCenters[clusterOffset+j]/=lenght;
			}
		}
	}
	private void randomizeClusters() {
		for (int i = 0; i < clustersCenters.length; i++) {
			clustersCenters[i] = (float) (Math.random()*256);
		}
	}

	private void prepareOpenCl(){
		int dimImageX = images.get(0).imageX;
		int dimImageY = images.get(0).imageY;

		
		Map<String, Object> params = new HashMap<>();
		params.put("IMAGE_SIZE", imageSize);
		params.put("NO_IMAGES", batchItems);
		params.put("FILTER_SIZE", dimFilter*dimFilter);
		params.put("NO_CLUSTERS", noClusters);
		params.put("DIM_FILTER", dimFilter);
		params.put("DIM_IMAGE_X", dimImageX);
		params.put("DIM_IMAGE_Y", dimImageY);
		params.put("STRIDE", stride);	
		program = new Program(Program.readResource("/dot/OnlineUpdateFilters.c"),params);
		
		memClusters = new MemoryFloat(program);
		memClusters.addReadWrite(clustersCenters);
		memImages = new MemoryFloat(program);
		memImages.addReadOnly(inputImages);
		
		updateCenters = new Kernel(program, "onlineUpdate");
		updateCenters.setArguments(memImages, memClusters);
	}
	public void releaseOpenCl(){
		memClusters.release();
		memImages.release();
		updateCenters.release();
		program.release();
	}
	private void constructImageClusters(){
		for (int i=0;i<noClusters;i++) {
			Image image = new ImageFloat(dimFilter,dimFilter);
			double max = -1*Double.MAX_VALUE;
			double min = Double.MAX_VALUE;
			int clusterOffset = dimFilter*dimFilter*i;
			for (int j = 0; j < dimFilter*dimFilter; j++) {
				if (clustersCenters[clusterOffset+j]>max)max=clustersCenters[clusterOffset+j];
				if (clustersCenters[clusterOffset+j]<min)min=clustersCenters[clusterOffset+j];
			}
			max=255/(max-min);
			for (int j = 0; j < dimFilter*dimFilter; j++) {
				image.getDataFloat()[j]=(float) ((clustersCenters[clusterOffset+j]-min)*max);
			}
			clusterImages.add(image);
		}			
	}
	public void constructNormalizedImageClusters(){
		clusterImages.clear();
		for (int i=0;i<noClusters;i++) {
			Image image = new ImageFloat(dimFilter,dimFilter);
			System.arraycopy(memClusters.getSrc(), i*dimFilter*dimFilter, image.getDataFloat(), 0, dimFilter*dimFilter);
			clusterImages.add(image);
		}			
	}
}
