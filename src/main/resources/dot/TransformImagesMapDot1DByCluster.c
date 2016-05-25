__kernel void transform(__global float const *images,__global float const *filters,__global float *transformedImages)
{
	double sum;
	int globalIndex=get_global_id(0);
	int imagesOffset = globalIndex*IMAGE_SIZE;
	int dimTransX = ((DIM_IMAGE_X - DIM_FILTER)/STRIDE+1);
	int dimTransY = ((DIM_IMAGE_Y - DIM_FILTER)/STRIDE+1);
	int transImagesOffset = globalIndex*dimTransY*dimTransX*NO_CLUSTERS;
	int imageX;
	int imageY;
	int filterX;
	int filterY;
	int index;
	int centerIndex;
	int transX=0;
	int transY=0;
	float subImageBuffer[FILTER_SIZE];

	for(imageY=0;imageY<=DIM_IMAGE_Y-DIM_FILTER;imageY+=STRIDE)
	{
		transX=0;
		for(imageX=0;imageX<=DIM_IMAGE_X-DIM_FILTER;imageX+=STRIDE)
		{
			index=0;
			for(filterY=0;filterY<DIM_FILTER;filterY++)
			{
				for(filterX=0;filterX<DIM_FILTER;filterX++)
				{
					subImageBuffer[index++] = images[imagesOffset+(imageY+filterY)*DIM_IMAGE_X+(imageX+filterX)];
				}
			}
			for(centerIndex=0;centerIndex<NO_CLUSTERS;centerIndex++)
			{
				sum = 0;
				for(index=0;index<FILTER_SIZE;index++)
				{
					sum+= filters[centerIndex*FILTER_SIZE+index]*subImageBuffer[index];
				}
				if(sum>0)
					transformedImages[transImagesOffset+transY*dimTransX*NO_CLUSTERS+transX+centerIndex*dimTransX]=sum/4;
				else
					transformedImages[transImagesOffset+transY*dimTransX*NO_CLUSTERS+transX+centerIndex*dimTransX]=0;
			}
			transX++;
		}
		transY++;
	}
}
