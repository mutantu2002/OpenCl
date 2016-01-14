#define IMAGE_SIZE  16
#define NO_CLUSTERS  256
#define IMAGES_PER_WORK_ITEM  625
#define WORK_ITEMS 2560

__kernel void updateCenters(__global double *centers, __global double *images, __global double *updates)
{
	int gid = get_global_id(0);
	
	int imagesOffset = gid*IMAGE_SIZE*IMAGES_PER_WORK_ITEM;
	int updatesOffset = gid*(IMAGE_SIZE+1)*NO_CLUSTERS;

	int centersIndex=0;
	int imageIndex=0;
	
	double sum=0;
	int index=0;
	double weight;
	double min;
	int minCenterIndex;
	double imageBuffer[IMAGE_SIZE];
	
	for(imageIndex=0;imageIndex<IMAGES_PER_WORK_ITEM;imageIndex++)
	{
		min=IMAGE_SIZE*1000000;
		minCenterIndex=0;
		for(centersIndex=0;centersIndex<NO_CLUSTERS;centersIndex++)
		{
			sum = 0;
			for(index=0;index<IMAGE_SIZE;index++)
			{
				imageBuffer[index]=images[imagesOffset+imageIndex*IMAGE_SIZE+index];
				weight = centers[centersIndex*IMAGE_SIZE+index]-imageBuffer[index];
				sum = sum+weight*weight;
			}
			if (sum<min)
			{
				min = sum;
				minCenterIndex = centersIndex;
			}
		}
		for(index=0;index<IMAGE_SIZE;index++)
		{
			updates[updatesOffset+(IMAGE_SIZE+1)*minCenterIndex+index] = updates[updatesOffset+(IMAGE_SIZE+1)*minCenterIndex+index]+imageBuffer[index];
		}
		updates[updatesOffset+(IMAGE_SIZE+1)*minCenterIndex+IMAGE_SIZE] = updates[updatesOffset+(IMAGE_SIZE+1)*minCenterIndex+IMAGE_SIZE]+1;
	}
}

__kernel void reduceCenters(__global double *centers,  __global double *updates)
{
	int offsetCenter = get_global_id(0);
	int indexWorkItem=0;
	double sum;
	double centerBuffer[IMAGE_SIZE+1];
	int centerIndex;
	for(centerIndex=0;centerIndex<IMAGE_SIZE+1;centerIndex++)
	{
		centerBuffer[centerIndex]=0;
	}
	for(indexWorkItem=0;indexWorkItem<WORK_ITEMS;indexWorkItem++)
	{
		for(centerIndex=0;centerIndex<IMAGE_SIZE+1;centerIndex++)
		{
			centerBuffer[centerIndex]=centerBuffer[centerIndex]+updates[(indexWorkItem*NO_CLUSTERS+offsetCenter)*(IMAGE_SIZE+1)+centerIndex];
		}
	}
	if (centerBuffer[IMAGE_SIZE]>0){
		for(centerIndex=0;centerIndex<IMAGE_SIZE;centerIndex++)
		{
			centers[offsetCenter*IMAGE_SIZE+centerIndex]=centerBuffer[centerIndex]/centerBuffer[IMAGE_SIZE];
		}
	}
}