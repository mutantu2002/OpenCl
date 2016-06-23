__kernel void train(__global const float *perceptrons, __global const short *images, __global float *updates )
{
	int imagesOffset = get_global_id(0)*IMAGE_SIZE*BATCH_ITEMS;
	int updatesOffset = get_global_id(0)*NO_PERCEPTRONS*IMAGE_SIZE;

	int perceptron;
	float sum;
	int offset;
	int batch;
	float activationOld[NO_PERCEPTRONS];
	float activation;

	for(batch=0;batch<BATCH_ITEMS;batch++)
	{
		for(perceptron=0;perceptron<NO_PERCEPTRONS;perceptron++)
		{
			sum=0;
			for(offset=0;offset<IMAGE_SIZE;offset++)
			{
				sum+=perceptrons[perceptron*IMAGE_SIZE+offset]*images[imagesOffset+batch*IMAGE_SIZE+offset];
			}
			sum/=IMAGE_SIZE;
			activation = sum-activationOld[perceptron];
			//activation=fabs(activation);
			activationOld[perceptron]=sum;
			if(batch==0) continue;
			for(offset=0;offset<IMAGE_SIZE;offset++)
			{
				updates[updatesOffset+perceptron*IMAGE_SIZE+offset]+=-images[imagesOffset+batch*IMAGE_SIZE+offset]*activation;
			}
		}
	}
}
__kernel void forward(__global const float *perceptrons, __global const short *images, __global float *activations )
{
	int imagesOffset = get_global_id(0)*IMAGE_SIZE;
	int activationsOffset = get_global_id(0)*NO_PERCEPTRONS;
	float sum;
	int perceptron;
	int offset;

	for(perceptron=0;perceptron<NO_PERCEPTRONS;perceptron++)
	{
		sum=0;
		for(offset=0;offset<IMAGE_SIZE;offset++)
		{
			sum+=perceptrons[perceptron*IMAGE_SIZE+offset]*images[imagesOffset+offset];
		}
		sum/=IMAGE_SIZE;
		activations[activationsOffset+perceptron]=sum;
	}
}
