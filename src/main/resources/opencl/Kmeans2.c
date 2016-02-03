__kernel void updateCenters(__global float *centers, __global const float *images, __global int *updates, int noClusters)
{
	int gid = get_global_id(0);
	int imagesOffset;
	int centersIndex=0;
	
	float sum=0;
	int index=0;
	float weight;
	float min;
	int minCenterIndex=-1;
	float pImage[imageSize];
	
	imagesOffset = gid*imageSize;
	for (index=0;index<imageSize;index++){
		pImage[index]=images[imagesOffset+index];
	}
	
	
	min=FLT_MAX;
	for(centersIndex=0;centersIndex<noClusters;centersIndex++)
	{
		sum = 0;
		for(index=0;index<imageSize;index++)
		{
			weight = centers[centersIndex*imageSize+index]-pImage[index];
			sum = sum+weight*weight;
		}
		if (sum<min)
		{
			min = sum;
			minCenterIndex = centersIndex;
		}
	}
	updates[gid]=minCenterIndex;
}

__kernel void reduceCenters(__global float *centers, __global const float *images, __global int *updates, const int noImages)
{
	int offsetCenter = get_global_id(0);
	int centerIndex;
	int indexImage;
	int noMembers=0;
	int imageOffset;
	int offsetSizeCenter=offsetCenter*imageSize;
	
	for(indexImage=0;indexImage<noImages;indexImage++)
	{
		if(updates[indexImage]==offsetCenter)
		{
			noMembers=noMembers+1;
			imageOffset = indexImage*imageSize;
			for(centerIndex=0;centerIndex<imageSize;centerIndex++)
			{
				centers[offsetSizeCenter+centerIndex]+=images[imageOffset+centerIndex];
			}
		}
	}
	if (noMembers>0)
	{
		for(centerIndex=0;centerIndex<imageSize;centerIndex++)
		{
			centers[offsetSizeCenter+centerIndex]/=noMembers;
		}
	}
}
