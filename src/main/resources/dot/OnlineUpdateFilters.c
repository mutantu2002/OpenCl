__kernel void onlineUpdate(__global float const *images,__global float *filters)
{
	double sum;
	double lenght;
	int filterOffset = get_global_id(0)*FILTER_SIZE;

	int imageX;
	int imageY;
	int filterX;
	int filterY;
	int index;
	int indexImage;
	float filterBuffer[FILTER_SIZE];
	int noUpdates=0;
	int noPass=0;
	int maxX;
	int maxY;
	int max=0;
	int maxImage;

	for(index=0;index<FILTER_SIZE;index++)
	{
		filterBuffer[index]=0;
	}
	for(indexImage=0;indexImage<NO_IMAGES;indexImage++)
	{
		for(imageY=0;imageY<=DIM_IMAGE_Y-DIM_FILTER;imageY+=STRIDE)
		{
			for(imageX=0;imageX<=DIM_IMAGE_X-DIM_FILTER;imageX+=STRIDE)
			{
				index=0;
				sum=0;
				for(filterY=0;filterY<DIM_FILTER;filterY++)
				{
					for(filterX=0;filterX<DIM_FILTER;filterX++)
					{
						sum+= filters[filterOffset+index]*images[indexImage*IMAGE_SIZE+(imageY+filterY)*DIM_IMAGE_X+(imageX+filterX)];
						index++;
					}
				}
				if(sum>max){
					maxX=imageX;
					maxY=imageY;
					maxImage=indexImage;
					max=sum;
				}
				noPass++;
				if(noPass>1000 && max>0)
				{
					index=0;
					for(filterY=0;filterY<DIM_FILTER;filterY++)
					{
						for(filterX=0;filterX<DIM_FILTER;filterX++)
						{
							filterBuffer[index]+= images[maxImage*IMAGE_SIZE+(maxY+filterY)*DIM_IMAGE_X+(maxX+filterX)];
							index++;
						}
					}
					noUpdates++;
					noPass=0;
					max=0;
					if(noUpdates>10){
						sum=0;
						for(index=0;index<FILTER_SIZE;index++)
						{
							sum+=filterBuffer[index];
						}
						sum=sum/FILTER_SIZE;
						lenght=0;
						for(index=0;index<FILTER_SIZE;index++)
						{
							filterBuffer[index]-=sum;
							lenght+=filterBuffer[index]*filterBuffer[index];
						}
						lenght=sqrt(lenght);
						for(index=0;index<FILTER_SIZE;index++)
						{
							filters[filterOffset+index]=filterBuffer[index]/lenght;
						}
						noUpdates=0;
						for(index=0;index<FILTER_SIZE;index++)
						{
							filterBuffer[index]=0;
						}
					}
				}
			}
		}
	}
}
