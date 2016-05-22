package home.mutant.opencl.dot.steps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageFloat;
import home.mutant.opencl.model.Kernel;
import home.mutant.opencl.model.MemoryFloat;
import home.mutant.opencl.model.Program;

public class ObtainFiltersDot {
	List<Image> images;
	int dimFilter;
	int imageSize;
	List<Image> clusterImages = new ArrayList<>();
	int noIterations;
	int noClusters;
	int batchItems = 256*3;
	int stride=1;
	float[] inputImages;
	float[] clustersCenters;
	float[] clustersUpdates;
	int stridePooling = 2;
	Program program;
	
	MemoryFloat memClusters;
	MemoryFloat memImages;
	MemoryFloat memUpdates;
	Kernel updateCenters;
	
	public ObtainFiltersDot(List<Image> images, int dimFilter, int noClusters, int noIterations) {
		this(images, dimFilter, noClusters, noIterations, 1);
	}

	public ObtainFiltersDot(List<Image> images, int dimFilter, int noClusters, int noIterations,int stride) {
		super();
		this.images = images;
		this.dimFilter = dimFilter;
		this.noIterations = noIterations;
		this.noClusters = noClusters;
		this.stride = stride;
		this.imageSize = images.get(0).getDataFloat().length;
		inputImages= new float[imageSize*batchItems];
		
		clustersCenters = new float[dimFilter*dimFilter*noClusters];
		randomizeClusters();
		subtractMeanClusters();
		clustersUpdates = new float[(dimFilter*dimFilter+1)*noClusters*batchItems];
	}
	public void cluster(){
		prepareOpenCl();
		for (int iteration=0;iteration<noIterations;iteration++){
			Arrays.fill(clustersUpdates, 0);
			memUpdates.copyHtoD();
			for (int batch=0 ;batch<images.size()/batchItems;batch++){
				for (int i=0;i<batchItems;i++){
					System.arraycopy(images.get(batch*batchItems+i).getDataFloat(), 0, inputImages, i*imageSize, imageSize);
				}
				memImages.copyHtoD();
				updateCenters.run(batchItems, 256);
				program.finish();
			}
			System.out.println(iteration);
			memUpdates.copyDtoH();
			reduceCenters();
			subtractMeanClusters();
			memClusters.copyHtoD();
		}
		releaseOpenCl();
		constructImageClusters();
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
			}else{
				System.out.println("Oops unused cluster");
			}
		}
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
	private void randomizeClusters2() {
		List<Image> filters = new ArrayList<>();
		int dimImage = (int)Math.sqrt(imageSize);
		while(filters.size()<noClusters) {
			Image initCluster = images.get((int) (Math.random()*images.size()));
			int x=stride*((int) (Math.random()*((dimImage-dimFilter)/stride)));
			int y=stride*((int) (Math.random()*((dimImage-dimFilter)/stride)));
			Image filter = initCluster.extractImage(x, y, dimFilter, dimFilter);
			if (okToAdd(filters, filter)){
				filters.add(filter);
			}
		}
		for (int i = 0; i < noClusters; i++) {
			System.arraycopy(filters.get(i).getDataFloat(), 0, clustersCenters, i*dimFilter*dimFilter, dimFilter*dimFilter);
		}
	}
	private boolean okToAdd(List<Image> filters, Image newFilter){
		for (Image image : filters) {
			if(d(image,newFilter)<100) return false;
		}
		return true;
	}
	private float d(Image i1, Image i2){
		double d=0;
		for (int i = 0; i < i1.getDataFloat().length; i++) {
			d+=(i1.getDataFloat()[i] - i2.getDataFloat()[i])*(i1.getDataFloat()[i] - i2.getDataFloat()[i]);
		}
		return (float) Math.sqrt(d);
	}
	private void prepareOpenCl(){
		int dimImage = (int)Math.sqrt(imageSize);
		int dimPooling = dimFilter+dimFilter;
		if(dimPooling>dimImage)dimPooling = dimImage;
		
		Map<String, Object> params = new HashMap<>();
		params.put("IMAGE_SIZE", imageSize);
		params.put("FILTER_SIZE", dimFilter*dimFilter);
		params.put("NO_CLUSTERS", noClusters);
		params.put("DIM_FILTER", dimFilter);
		params.put("DIM_POOLING", dimPooling);
		params.put("DIM_IMAGE", dimImage);
		params.put("STRIDE", stride);
		params.put("STRIDE_POOLING", stridePooling);
		
		program = new Program(Program.readResource("/dot/SubImageKmeansDotProduct.c"),params);
		
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
	private void constructImageClusters(){
		for (int i=0;i<noClusters;i++) {
			Image image = new ImageFloat(dimFilter*dimFilter);
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
}
