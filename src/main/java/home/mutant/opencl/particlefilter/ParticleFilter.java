package home.mutant.opencl.particlefilter;

import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageFloat;
import home.mutant.dl.utils.MathUtils;
import home.mutant.opencl.particlefilter.steps.ParticleFilterOpenCl;

public class ParticleFilter {
	public static final int NO_RND=1000;
	Image mapImage;
	Image inputImage;
	int dimMap;
	int dimImage;
	int imageSize;
	int mapSize;
	Particles particles;
	int noParticles;
	int currentNoParticles;
	int xInput;
	int yInput;
	int step=0;
	int[] rnd = new int[NO_RND];
	double sum=0;
	
	ParticleFilterOpenCl particleOpencl;
	
	public ParticleFilter(Image mapImage, Image inputImage, int noParticles){
		this.mapImage=mapImage;
		this.inputImage=inputImage;
		this.imageSize = inputImage.getDataFloat().length;
		this.dimImage = (int) Math.sqrt(imageSize);
		this.mapSize = mapImage.getDataFloat().length;
		this.dimMap = (int) Math.sqrt(mapSize);
		this.noParticles = noParticles;
		this.currentNoParticles = noParticles;
		xInput=0;//dimImage/2;
		yInput=0;//dimImage/2;
		for (int i=0;i<NO_RND;i++){
			rnd[i]=1-(int)(3*Math.random());
		}
		initParticles();
	}

	private void initParticles() {
		particles = new Particles(noParticles);
		for(int i=0;i<noParticles;i++){
			particles.x[i]=(int) (Math.random()*dimMap);
			particles.y[i]=(int) (Math.random()*dimMap);
			particles.weights[i]=1./noParticles;
		}
		particleOpencl=new ParticleFilterOpenCl(mapImage.getDataFloat(), particles.x, particles.y, particles.weights, rnd);
	}
	public void step(){
		float measurement = inputImage.getDataFloat()[yInput*dimImage+xInput];
		estimateParticlesWeightsOpenCl(measurement);
		normalizeWeights();
		recreateParticles();
		int xNew=xInput;
		int yNew=yInput;
		
//		boolean test;
//		do{
//			//xNew=(int) (dimImage*Math.random());
//			xNew = xInput+1-(int)(3*Math.random());
//			if(xNew>=dimImage)xNew=dimImage-1;
//			if(xNew<0)xNew=0;
//			//yNew=(int) (dimImage*Math.random());
//			yNew = yInput+1-(int)(3*Math.random());
//			if(yNew>=dimImage)yNew=dimImage-1;
//			if(yNew<0)yNew=0;
//			
//			if(step%2==0){
//				test=inputImage.getDataFloat()[yNew*dimImage+xNew]==0;
//			}else{
//				test=inputImage.getDataFloat()[yNew*dimImage+xNew]!=0;
//			}
//		}while(test);
		xNew=xInput+1;
		if(xNew>=dimImage){
			xNew=0;
			yNew=yInput+1;
			if(yNew>=dimImage){
				yNew=0;
			}
		}
		step++;
		moveParticlesOpenCl(xNew-xInput, yNew-yInput);
		xInput=xNew;
		yInput=yNew;
	}
	@SuppressWarnings("unused")
	private void estimateParticlesWeights(float measurement){
		for(int i=0;i<currentNoParticles;i++){
			float diff = mapImage.getDataFloat()[(particles.y[i]*dimMap+particles.x[i])]-measurement;
			particles.weights[i]=MathUtils.gaussian(diff, 40);
		}
	}
	private void estimateParticlesWeightsOpenCl(float measurement){
		particleOpencl.estimate(measurement, currentNoParticles);
	}
	private void normalizeWeights(){
		sum=0;
		for(int i=0;i<currentNoParticles;i++){
			sum+= particles.weights[i];
		}
	}
	
	private void recreateParticles(){
		particles.copyToTmp();
		int newIndex=0;
		//noParticles/=1.01;
		for(int i=0;i<currentNoParticles;i++){
			int noNew = (int) (particles.weights[i]/sum*noParticles);
			for(int newI=0;newI<noNew;newI++){
				particles.x[newIndex]=particles.xTmp[i];
				particles.y[newIndex]=particles.yTmp[i];
				newIndex++;
			}
		}
		currentNoParticles = newIndex;
	}
	
	private void recreateParticles2(){
		particles.copyToTmp();
		int newIndex=0;
		for(int i=0;i<currentNoParticles;i++){
			int noNew=0;
			if(particles.weights[i]>0.5){
				noNew=2;
				//System.out.println(noNew);
				for(int newI=0;newI<noNew;newI++){
					particles.x[newIndex]=particles.xTmp[i];
					particles.y[newIndex]=particles.yTmp[i];
					newIndex++;
				}
			}
		}
		currentNoParticles = newIndex;
	}
	
	private void moveParticlesOpenCl(int x, int y){
		int indexRnd = (int) (NO_RND * Math.random());
		particleOpencl.move(x, y, indexRnd,currentNoParticles);
		
	}
	@SuppressWarnings("unused")
	private void moveParticles(int x, int y){
		int index = (int) (NO_RND * Math.random());
		for(int i=0;i<currentNoParticles;i++){
			particles.x[i]+=x+rnd[index++];
			index%=NO_RND;
			if(particles.x[i]>=dimMap)particles.x[i]=dimMap-1;
			if(particles.x[i]<0)particles.x[i]=0;
			particles.y[i]+=y+rnd[index++];
			index%=NO_RND;
			if(particles.y[i]>=dimMap)particles.y[i]=dimMap-1;
			if(particles.y[i]<0)particles.y[i]=0;
		}
	}
	
	public Image getImageParticles(){
		Image imgParticles = new ImageFloat(mapImage.getDataFloat().length);
		for(int i=0;i<currentNoParticles;i++){
			double value = imgParticles.getPixel(particles.x[i], particles.y[i])+20;
			if(value>255)value=255;
			imgParticles.setPixel(particles.x[i], particles.y[i], value);
		}
		return imgParticles;
	}

	public void release() {
		particleOpencl.releaseOpenCl();
		
	}
}
