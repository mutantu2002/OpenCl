__kernel void output(__global float *inputs, __global int *labels, __global float *weights, __global float *outputs)
{
	int weightsOffset = get_global_id(0)*(INPUT_SIZE+1);
	int outputOffset = get_global_id(0)*10;
	int indexInput;
	int indexWeights;
	float sum;
	for(indexInput=0;indexInput<NO_INPUTS;indexInput++)
	{
		sum=0;
		for(indexWeights=0;indexWeights<INPUT_SIZE;indexWeights++)
		{
			sum+=inputs[indexInput*INPUT_SIZE+indexWeights]*weights[weightsOffset+indexWeights];
		}
		sum+=weights[weightsOffset+INPUT_SIZE];
		if(sum>15)
		{
			outputs[outputOffset+labels[indexInput]]+=1;
//			for(indexWeights=0;indexWeights<INPUT_SIZE;indexWeights++)
//			{
//				weights[weightsOffset+indexWeights]+=inputs[indexInput*INPUT_SIZE+indexWeights]*0.0004;
//			}
			//weights[weightsOffset+INPUT_SIZE]+=0.0004;
		}else
		{
//			for(indexWeights=0;indexWeights<INPUT_SIZE;indexWeights++)
//			{
//				weights[weightsOffset+indexWeights]-=inputs[indexInput*INPUT_SIZE+indexWeights]*0.0004;
//			}
//			weights[weightsOffset+INPUT_SIZE]-=0.0004;
		}
	}
}
