__kernel void meanPolling4D(__global float *inImages, __global float *outImages)
{
	int inImageOffset = get_global_id(0)*IN_IMAGE_SIZE;
	int outImagesOffset = get_global_id(0)*OUT_IMAGE_SIZE;
	
	float mean;
	
	int imageX;
	int imageY;
	
	for(imageX=0;imageX<IN_DIM_IMAGE/4;imageX++)
	{
		for(imageY=0;imageY<IN_DIM_IMAGE/4;imageY++)
		{
			mean=inImages[inImageOffset+(4*imageY)*IN_DIM_IMAGE+4*imageX];
			mean+=inImages[inImageOffset+(4*imageY+2)*IN_DIM_IMAGE+4*imageX];
			mean+=inImages[inImageOffset+(4*imageY)*IN_DIM_IMAGE+4*imageX+2];
			mean+=inImages[inImageOffset+(4*imageY+2)*IN_DIM_IMAGE+4*imageX+2];
			outImages[outImagesOffset+(2*imageY)*OUT_DIM_IMAGE+2*imageX]=mean/4;

			mean=inImages[inImageOffset+(4*imageY+1)*IN_DIM_IMAGE+4*imageX];
			mean+=inImages[inImageOffset+(4*imageY+3)*IN_DIM_IMAGE+4*imageX];
			mean+=inImages[inImageOffset+(4*imageY+1)*IN_DIM_IMAGE+4*imageX+2];
			mean+=inImages[inImageOffset+(4*imageY+3)*IN_DIM_IMAGE+4*imageX+2];
			outImages[outImagesOffset+(2*imageY+1)*OUT_DIM_IMAGE+2*imageX]=mean/4;

			mean=inImages[inImageOffset+(4*imageY)*IN_DIM_IMAGE+4*imageX+1];
			mean+=inImages[inImageOffset+(4*imageY+2)*IN_DIM_IMAGE+4*imageX+1];
			mean+=inImages[inImageOffset+(4*imageY)*IN_DIM_IMAGE+4*imageX+3];
			mean+=inImages[inImageOffset+(4*imageY+2)*IN_DIM_IMAGE+4*imageX+3];
			outImages[outImagesOffset+(2*imageY)*OUT_DIM_IMAGE+2*imageX+1]=mean/4;

			mean=inImages[inImageOffset+(4*imageY+1)*IN_DIM_IMAGE+4*imageX+1];
			mean+=inImages[inImageOffset+(4*imageY+3)*IN_DIM_IMAGE+4*imageX+1];
			mean+=inImages[inImageOffset+(4*imageY+1)*IN_DIM_IMAGE+4*imageX+3];
			mean+=inImages[inImageOffset+(4*imageY+3)*IN_DIM_IMAGE+4*imageX+3];
			outImages[outImagesOffset+(2*imageY+1)*OUT_DIM_IMAGE+2*imageX+1]=mean/4;
		}
	}
}
