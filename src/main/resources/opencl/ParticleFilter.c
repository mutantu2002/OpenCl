__kernel void estimateWeights(__global float const *map, __global int *x, __global int *y, __global double *weights, float measurement )
{
	int index=get_global_id(0);
	double w;
	w=map[y[index]*DIM_MAP+x[index]]-measurement;
	weights[index]=exp(-1*w*w/2/MEAN/MEAN);
}
