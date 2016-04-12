__kernel void transform(__global float const *images,__global float *filters,__global float *transformedImages)
{
	int transDimImage = DIM_NO_CLUSTERS*((DIM_IMAGE - DIM_FILTER)/STRIDE+1);
	double sum;
	float weight;
	int globalIndex=get_global_id(0);
	int imagesOffset = globalIndex*IMAGE_SIZE;
	int transImagesOffset = globalIndex*transDimImage*transDimImage;
	int imageX;
	int imageY;
	int filterX;
	int filterY;
	int index;
	int centersIndexX;
	int centersIndexY;
	int transX;
	int transY;
	
	float subImageBuffer[FILTER_SIZE];

	
	for(imageY=0;imageY<=DIM_IMAGE-DIM_FILTER;imageY+=STRIDE)
	{
		for(imageX=0;imageX<=DIM_IMAGE-DIM_FILTER;imageX+=STRIDE)
		{
			index=0;
			for(filterY=0;filterY<DIM_FILTER;filterY++)
			{
				for(filterX=0;filterX<DIM_FILTER;filterX++)
				{
					subImageBuffer[index++] = images[imagesOffset+(imageY+filterY)*DIM_IMAGE+(imageX+filterX)];
				}
			}
			for(centersIndexX=0;centersIndexX<DIM_NO_CLUSTERS;centersIndexX++)
			{
				for(centersIndexY=0;centersIndexY<DIM_NO_CLUSTERS;centersIndexY++)
				{
					sum = 0;
					for(index=0;index<FILTER_SIZE;index++)
					{
						weight = filters[(centersIndexY*DIM_NO_CLUSTERS+centersIndexX)*FILTER_SIZE+index]-subImageBuffer[index];
						sum = sum+weight*weight;
					}
					transY=imageY/STRIDE*DIM_NO_CLUSTERS+centersIndexY;
					transX=imageX/STRIDE*DIM_NO_CLUSTERS+centersIndexX;
					transformedImages[transImagesOffset+transY*transDimImage+transX]=exp(-1*sum/2/MEAN/MEAN)*255;
				}
			}
		}
	}

}
