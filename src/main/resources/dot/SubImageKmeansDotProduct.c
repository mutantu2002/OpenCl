__kernel void updateCenters(__global float *centers, __global const float *images, __global float *updates, __local float *centersLocal )
{
	int imagesOffset = get_global_id(0)*IMAGE_SIZE;
	
	int updatesOffset = get_global_id(0)*(FILTER_SIZE+1)*NO_CLUSTERS;
	int centersIndex=0;
	
	float sum=0;
	int index=0;
	float weight;
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
	
	for(imagePX=0;imagePX<=DIM_IMAGE-DIM_POOLING;imagePX+=STRIDE_POOLING)
	{
		for(imagePY=0;imagePY<=DIM_IMAGE-DIM_POOLING;imagePY+=STRIDE_POOLING)
		{
			max=-FLT_MAX;
			maxCenterIndex=0;

			for(imageX=0;imageX<=DIM_POOLING-DIM_FILTER;imageX+=STRIDE)
			{
				for(imageY=0;imageY<=DIM_POOLING-DIM_FILTER;imageY+=STRIDE)
				{
					index=0;
					for(filterX=0;filterX<DIM_FILTER;filterX++)
					{
						for(filterY=0;filterY<DIM_FILTER;filterY++)
						{
							subImageBuffer[index++] = images[imagesOffset+(imageY+imagePY+filterY)+(imageX+imagePX+filterX)*DIM_IMAGE];
						}
					}

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
			for(filterX=0;filterX<DIM_FILTER;filterX++)
			{
				for(filterY=0;filterY<DIM_FILTER;filterY++)
				{
					updates[updatesOffset+maxCenterIndex+index++]+= images[imagesOffset+(maxY+filterY)+(maxX+filterX)*DIM_IMAGE];
				}
			}
			updates[updatesOffset+maxCenterIndex+FILTER_SIZE]+=1;
		}
	}
}
