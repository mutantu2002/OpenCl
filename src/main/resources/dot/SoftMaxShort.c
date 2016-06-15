__kernel void train(__global const float *perceptrons, __global const short *images, __global float *updates,  __global const short *labels )
{
	int imagesOffset = get_global_id(0)*IMAGE_SIZE*BATCH_ITEMS;
	int updatesOffset = get_global_id(0)*NO_CLASSES*(IMAGE_SIZE+1);
	int labelsOffset = get_global_id(0)*BATCH_ITEMS;

	int perceptron;
	float sum;
	float maxForExp;
	int offset;
	int batch;
	float activation[NO_CLASSES];
	for(batch=0;batch<BATCH_ITEMS;batch++)
	{
		maxForExp = -FLT_MAX;
		for(perceptron=0;perceptron<NO_CLASSES;perceptron++)
		{
			sum=0;
			for(offset=0;offset<IMAGE_SIZE;offset++)
			{
				sum+=perceptrons[perceptron*(IMAGE_SIZE+1)+offset]*images[imagesOffset+batch*IMAGE_SIZE+offset]/128.;
			}
			sum+=perceptrons[perceptron*(IMAGE_SIZE+1)+IMAGE_SIZE];
			if(sum>maxForExp)
			{
				maxForExp=sum;
			}
			activation[perceptron]=sum;
		}
		for(perceptron=0;perceptron<NO_CLASSES;perceptron++)
		{
			activation[perceptron]-=maxForExp;
			activation[perceptron]=exp(activation[perceptron]);
		}

		sum=0;
		for(perceptron=0;perceptron<NO_CLASSES;perceptron++)
		{
			sum+=activation[perceptron];
		}

		for(perceptron=0;perceptron<NO_CLASSES;perceptron++)
		{
			if(perceptron==labels[labelsOffset+batch])
			{
				activation[perceptron]=activation[perceptron]/sum-1;
			}else
			{
				activation[perceptron]/=sum;
			}
		}

		for(perceptron=0;perceptron<NO_CLASSES;perceptron++)
		{
			for(offset=0;offset<IMAGE_SIZE;offset++)
			{
				updates[updatesOffset+perceptron*(IMAGE_SIZE+1)+offset]+=-images[imagesOffset+batch*IMAGE_SIZE+offset]*activation[perceptron]/128.;
			}
			updates[updatesOffset+perceptron*(IMAGE_SIZE+1)+IMAGE_SIZE]+=-activation[perceptron];
		}
	}
}
__kernel void test(__global const float *perceptrons, __global const short *images, __global short *labels )
{
	int imagesOffset = get_global_id(0)*IMAGE_SIZE;
	int labelsOffset = get_global_id(0);
	float sum;
	int perceptron;
	int offset;
	float max=-FLT_MAX;
	int maxLabel;

	for(perceptron=0;perceptron<NO_CLASSES;perceptron++)
	{
		sum=0;
		for(offset=0;offset<IMAGE_SIZE;offset++)
		{
			sum+=perceptrons[perceptron*(IMAGE_SIZE+1)+offset]*images[imagesOffset+offset]/128.;
		}
		sum+=perceptrons[perceptron*(IMAGE_SIZE+1)+IMAGE_SIZE];
		if(sum>max)
		{
			max=sum;
			maxLabel=perceptron;
		}
	}
	labels[labelsOffset]=maxLabel;
}
