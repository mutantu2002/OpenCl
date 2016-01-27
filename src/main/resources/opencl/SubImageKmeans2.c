__kernel void updateCenters(__global float *centers, __global const float *images, __global float *updates)
{
	int imagesOffset = get_global_id(0)*IMAGE_SIZE;
	
	int updatesOffset = get_global_id(0)*(FILTER_SIZE+1)*NO_CLUSTERS;
	int centersIndex=0;
	
	float sum=0;
	int index=0;
	float weight;
	float min;
	int minCenterIndex;
	
	int imageX;
	int imageY;
	int filterX;
	int filterY;
	
	float subImageBuffer[FILTER_SIZE];

	for(imageX=0;imageX<=DIM_IMAGE-DIM_FILTER;imageX+=STRIDE)
	{
		for(imageY=0;imageY<=DIM_IMAGE-DIM_FILTER;imageY+=STRIDE)
		{
			index=0;
			for(filterX=0;filterX<DIM_FILTER;filterX++)
			{
				for(filterY=0;filterY<DIM_FILTER;filterY++)
				{
					subImageBuffer[index++] = images[imagesOffset+(imageY+filterY)+(imageX+filterX)*DIM_IMAGE];
				}
			}
			min=FLT_MAX;
			minCenterIndex=0;
			for(centersIndex=0;centersIndex<NO_CLUSTERS;centersIndex++)
			{
				sum = 0;
				for(index=0;index<FILTER_SIZE;index++)
				{
					weight = centers[centersIndex*FILTER_SIZE+index]-subImageBuffer[index];
					sum = sum+weight*weight;
				}
				if (sum<min)
				{
					min = sum;
					minCenterIndex = centersIndex;
				}
			}
			minCenterIndex = (FILTER_SIZE+1)*minCenterIndex;
			for(index=0;index<FILTER_SIZE;index++)
			{
				updates[updatesOffset+minCenterIndex+index]+= subImageBuffer[index];
			}
			updates[updatesOffset+minCenterIndex+FILTER_SIZE]=updates[updatesOffset+minCenterIndex+FILTER_SIZE]+1;
		}
	}
}
