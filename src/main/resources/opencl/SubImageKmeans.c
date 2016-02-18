#define DIM_FILTER  16
#define NO_CLUSTERS 64
#define DIM_IMAGE  84
#define IMAGE_SIZE  7056
#define FILTER_SIZE  (DIM_FILTER*DIM_FILTER)

#define INFLUENCE 1
#define STRIDE 8
#define WORK_ITEMS 256*39

__kernel void updateCenters(__global float *centers, __global float *images, __global float *updates)
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
			updates[updatesOffset+minCenterIndex+FILTER_SIZE]+= 1;
		}
	}
}

__kernel void reduceCenters(__global float *updates)
{
	int offsetCenter = get_global_id(0);
	int indexWorkItem=0;
	
	float centerBuffer[FILTER_SIZE+1];
	int centerIndex;
	for(centerIndex=0;centerIndex<FILTER_SIZE+1;centerIndex++)
	{
		centerBuffer[centerIndex]=0;
	}
	for(indexWorkItem=0;indexWorkItem<WORK_ITEMS;indexWorkItem++)
	{
		for(centerIndex=0;centerIndex<FILTER_SIZE+1;centerIndex++)
		{
			centerBuffer[centerIndex]=centerBuffer[centerIndex]+updates[(indexWorkItem*NO_CLUSTERS+offsetCenter)*(FILTER_SIZE+1)+centerIndex];
		}
	}
	if (centerBuffer[FILTER_SIZE]>0)
	{
		for(centerIndex=0;centerIndex<FILTER_SIZE;centerIndex++)
		{
			updates[offsetCenter*(FILTER_SIZE+1)+centerIndex]=centerBuffer[centerIndex]/centerBuffer[FILTER_SIZE];
		}
	}
}

__kernel void mixCenters2D(__global float *centers,  __global float *updates, const int dimNoClusters)
{
	int offsetCenter = get_global_id(0);
	int centerIndex;
	float influence=0;
	
	int offsetCenterX=offsetCenter%dimNoClusters;
	int offsetCenterY=offsetCenter/dimNoClusters;
	
	int offsetCenterX1=(offsetCenterX+1)%dimNoClusters;
	int offsetCenterY1=(offsetCenterY+1)%dimNoClusters;
	int offsetCenterX_1=(offsetCenterX+dimNoClusters-1)%dimNoClusters;
	int offsetCenterY_1=(offsetCenterY+dimNoClusters-1)%dimNoClusters;
	
	for(centerIndex=0;centerIndex<FILTER_SIZE;centerIndex++)
	{
		influence = updates[(offsetCenterY*dimNoClusters+offsetCenterX1)*(FILTER_SIZE+1)+centerIndex];

		influence = influence + updates[(offsetCenterY*dimNoClusters+offsetCenterX_1)*(FILTER_SIZE+1)+centerIndex];

		influence = influence + updates[(offsetCenterY1*dimNoClusters+offsetCenterX)*(FILTER_SIZE+1)+centerIndex];

		influence = influence + updates[(offsetCenterY_1*dimNoClusters+offsetCenterX)*(FILTER_SIZE+1)+centerIndex];

		centers[offsetCenter*FILTER_SIZE+centerIndex]=(INFLUENCE*updates[offsetCenter*(FILTER_SIZE+1)+centerIndex]+influence)/(INFLUENCE+4);
	}
}
