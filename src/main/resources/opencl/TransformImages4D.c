__kernel void transform(__global double const *images,__global double *filters,__global double *x, __global double *y, __global double *z, __global double *w,
		__global double *transformedImages)
{
	int transDimImage = 2*((DIM_IMAGE - DIM_FILTER)/STRIDE+1);
	float min;
	int minCenterIndex;
	int sum;
	int weight;
	int globalIndex=get_global_id(0);
	int imagesOffset = globalIndex*IMAGE_SIZE;
	int transImagesOffset = globalIndex*transDimImage*transDimImage;
	int imageX;
	int imageY;
	int filterX;
	int filterY;
	int transX=0;
	int transY=0;
	int index;
	int centersIndex;

	float subImageBuffer[FILTER_SIZE];

	for(imageX=0;imageX<=DIM_IMAGE-DIM_FILTER;imageX+=STRIDE)
	{
		transY=0;
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
			min=DBL_MAX;
			minCenterIndex=0;
			for(centersIndex=0;centersIndex<NO_CLUSTERS;centersIndex++)
			{
				sum = 0;
				for(index=0;index<FILTER_SIZE;index++)
				{
					weight = filters[centersIndex*FILTER_SIZE+index]-subImageBuffer[index];
					sum = sum+weight*weight;
				}
				if (sum<min)
				{
					min = sum;
					minCenterIndex = centersIndex;
				}
			}
			transformedImages[transImagesOffset+2*transY+(2*transX)*transDimImage]=x[minCenterIndex]+100;
			transformedImages[transImagesOffset+2*transY+1+(2*transX)*transDimImage]=y[minCenterIndex]+100;
			transformedImages[transImagesOffset+2*transY+(2*transX+1)*transDimImage]=z[minCenterIndex]+100;
			transformedImages[transImagesOffset+2*transY+1+(2*transX+1)*transDimImage]=w[minCenterIndex]+100;
			transY++;
		}
		transX++;
	}

}
