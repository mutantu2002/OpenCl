__kernel void transform(__global short const *images,__global float const *filters,__global short *transformedImages)
{
	double sum;
	int globalIndex=get_global_id(0);
	int imagesOffset = globalIndex*IMAGE_SIZE;
	int dimTransX = ((DIM_IMAGE_X - DIM_FILTER_X)/STRIDE_X+1);
	int dimTransY = ((DIM_IMAGE_Y - DIM_FILTER_Y)/STRIDE_Y+1);
	int transImagesOffset = globalIndex*dimTransY*dimTransX*NO_CLUSTERS;
	int imageX;
	int imageY;
	int filterX;
	int filterY;
	int index;
	int centerIndex;
	int transX=0;
	int transY=0;
	short subImageBuffer[FILTER_SIZE];

	for(imageY=0;imageY<=DIM_IMAGE_Y-DIM_FILTER_Y;imageY+=STRIDE_Y)
	{
		transX=0;
		for(imageX=0;imageX<=DIM_IMAGE_X-DIM_FILTER_X;imageX+=STRIDE_X)
		{
			index=0;
			for(filterY=0;filterY<DIM_FILTER_Y;filterY++)
			{
				for(filterX=0;filterX<DIM_FILTER_X;filterX++)
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
					transformedImages[transImagesOffset+(transY+centerIndex*dimTransY)*dimTransX+transX]=sum;
				else
					transformedImages[transImagesOffset+(transY+centerIndex*dimTransY)*dimTransX+transX]=0;
			}
			transX++;
		}
		transY++;
	}
}
