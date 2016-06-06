__kernel void maxPolling(__global short *inImages, __global short *outImages)
{
	int inImageOffset = get_global_id(0)*IN_IMAGE_SIZE;
	int outImagesOffset = get_global_id(0)*OUT_IMAGE_SIZE;
	
	int max;
	int current;
	int imageX;
	int imageY;
	
	for(imageX=0;imageX<IN_DIM_IMAGE/2;imageX++)
	{
		for(imageY=0;imageY<IN_DIM_IMAGE/2;imageY++)
		{
			max=inImages[inImageOffset+(2*imageY)*IN_DIM_IMAGE+2*imageX];
			current = inImages[inImageOffset+(2*imageY)*IN_DIM_IMAGE+2*imageX+1];
			if(max<current)
			{
				max=current;
			}
			current = inImages[inImageOffset+(2*imageY+1)*IN_DIM_IMAGE+2*imageX];
			if(max<current)
			{
				max=current;
			}
			current = inImages[inImageOffset+(2*imageY+1)*IN_DIM_IMAGE+2*imageX+1];
			if(max<current)
			{
				max=current;
			}
			outImages[outImagesOffset+imageY*OUT_DIM_IMAGE+imageX]=max;
		}
	}
}
