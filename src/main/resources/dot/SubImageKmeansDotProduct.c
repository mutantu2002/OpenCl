__kernel void updateCenters(__global const float *centers, __global const float *images, __global float *updates )
{
	int imagesOffset = get_global_id(0)*IMAGE_SIZE;
	
	int updatesOffset = get_global_id(0)*(FILTER_SIZE+1)*NO_CLUSTERS;
	int centersIndex=0;
	
	float sum=0;
	int index=0;
	float max;
	int maxCenterIndex;
	
	int imageX;
	int imageY;
	int imagePX;
	int imagePY;
	int filterX;
	int filterY;
	int maxX=0;
	int maxY=0;
	float subImageBuffer[FILTER_SIZE];
	bool okToProcess;
/*
	__local float centersLocal[NO_CLUSTERS*FILTER_SIZE];
	if(get_local_id(0)<NO_CLUSTERS)
	{
		for(index=0;index<FILTER_SIZE;index++)
		{
			centersLocal[get_local_id(0)*FILTER_SIZE+index]=centers[get_local_id(0)*FILTER_SIZE+index];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
*/
	for(imagePX=0;imagePX<=DIM_IMAGE_X-DIM_POOLING_X;imagePX+=STRIDE_POOLING_X)
	{
		for(imagePY=0;imagePY<=DIM_IMAGE_Y-DIM_POOLING_Y;imagePY+=STRIDE_POOLING_Y)
		{
			max=-FLT_MAX;
			maxCenterIndex=0;

			for(imageX=0;imageX<=DIM_POOLING_X-DIM_FILTER_X;imageX+=STRIDE_X)
			{
				for(imageY=0;imageY<=DIM_POOLING_Y-DIM_FILTER_Y;imageY+=STRIDE_Y)
				{
					index=0;
					okToProcess=false;
					for(filterY=0;filterY<DIM_FILTER_Y;filterY++)
					{
						for(filterX=0;filterX<DIM_FILTER_X;filterX++)
						{
							subImageBuffer[index] = images[imagesOffset+(imageY+imagePY+filterY)*DIM_IMAGE_X+(imageX+imagePX+filterX)];
							if(subImageBuffer[index]!=0)okToProcess=true;
							index++;
						}
					}
					if(!okToProcess) continue;
					for(centersIndex=0;centersIndex<NO_CLUSTERS;centersIndex++)
					{
						sum = 0;
						for(index=0;index<FILTER_SIZE;index++)
						{
							sum +=centers[centersIndex*FILTER_SIZE+index]*subImageBuffer[index];
						}
						if (sum>max)
						{
							max = sum;
							maxCenterIndex = centersIndex;
							maxX=imageX+imagePX;
							maxY=imageY+imagePY;
						}
					}

				}
			}

			index=0;
			maxCenterIndex = (FILTER_SIZE+1)*maxCenterIndex;
			for(filterY=0;filterY<DIM_FILTER_Y;filterY++)
			{
				for(filterX=0;filterX<DIM_FILTER_X;filterX++)
				{
					updates[updatesOffset+maxCenterIndex+index++]+= images[imagesOffset+(maxY+filterY)*DIM_IMAGE_X+(maxX+filterX)];
				}
			}
			updates[updatesOffset+maxCenterIndex+FILTER_SIZE]+=1;
		}
	}
}
