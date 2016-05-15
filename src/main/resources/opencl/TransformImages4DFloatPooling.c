__kernel void transform(__global float const *images,__global float *filters,__global float *x, __global float *y, __global float *z, __global float *w,
		__global float *transformedImages)
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
	int imagePX;
	int imagePY;
	float subImageBuffer[FILTER_SIZE];

	for(imagePX=0;imagePX<=DIM_IMAGE-DIM_POOLING;imagePX+=STRIDE_POOLING)
	{
		for(imagePY=0;imagePY<=DIM_IMAGE-DIM_POOLING;imagePY+=STRIDE_POOLING)
		{
			min=FLT_MAX;
			minCenterIndex=0;
			
			for(imageX=0;imageX<=DIM_POOLING-DIM_FILTER;imageX+=STRIDE)
			{
				transY=0;
				for(imageY=0;imageY<=DIM_POOLING-DIM_FILTER;imageY+=STRIDE)
				{
					index=0;
					for(filterX=0;filterX<DIM_FILTER;filterX++)
					{
						for(filterY=0;filterY<DIM_FILTER;filterY++)
						{
							subImageBuffer[index++] = images[imagesOffset+(imageY+filterY)+(imageX+filterX)*DIM_IMAGE];
						}
					}

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

				}
				
			}
			transformedImages[transImagesOffset+2*transY+(2*transX)*transDimImage]=x[minCenterIndex]+127;
			transformedImages[transImagesOffset+2*transY+1+(2*transX)*transDimImage]=y[minCenterIndex]+127;
			transformedImages[transImagesOffset+2*transY+(2*transX+1)*transDimImage]=z[minCenterIndex]+127;
			transformedImages[transImagesOffset+2*transY+1+(2*transX+1)*transDimImage]=w[minCenterIndex]+127;
			transY++;
		}
		transX++;
	}
}
