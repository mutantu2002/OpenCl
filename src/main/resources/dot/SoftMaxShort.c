__kernel void train(__global const float *perceptrons, __global const short *images, __global float *updates,  __global int *labels )
{
	int imagesOffset = get_global_id(0)*IMAGE_SIZE*BATCH_SIZE;
	int updatesOffset = get_global_id(0)*NO_CLASSES*(IMAGE_SIZE+1);
	int labelsOffset = get_global_id(0)*BATCH_SIZE;

	int batch;
	float activation[NO_CLASSES];
	for(batch=0;batch<BATCH_SIZE;batch++)
	{
		for(perceptron=0;perceptron<NO_CLASSES;perceptron++)
		{
			sum=0;
			for(offset=0;offset<IMAGE_SIZE;offset++)
			{
				sum+=perceptrons[perceptron*(IMAGE_SIZE+1)+offset]*images[imagesOffset+batch*IMAGE_SIZE+offset];
			}
			activation[perceptron]=exp(sum);
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
				updates[updatesOffset+perceptron*(IMAGE_SIZE+1)+offset]+=-images[imagesOffset+batch*IMAGE_SIZE+offset]*activation[perceptron];
			}
			updates[updatesOffset+perceptron*(IMAGE_SIZE+1)+IMAGE_SIZE]+=-activation[perceptron];
		}
	}
}
