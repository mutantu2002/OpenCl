__kernel void estimateWeights(__global float const *map, __global int *x, __global int *y, __global double *weights, float measurement )
{
	int index=get_global_id(0);
	double w;
	w=map[y[index]*DIM_MAP+x[index]]-measurement;
	weights[index]=exp(-1*w*w/2/MEAN/MEAN);
}
__kernel void moveParticles(__global int *x, __global int *y, __global int const *rnd, int indexRnd, int dx, int dy )
{
	int index=get_global_id(0);
	int i = indexRnd+2*index;
	i=i%NO_RND;
	x[index]+=dx+rnd[i];
	if(x[index]>=DIM_MAP)x[index]=DIM_MAP-1;
	if(x[index]<0)x[index]=0;
	i++;
	i=i%NO_RND;
	y[index]+=dy+rnd[i];
	if(y[index]>=DIM_MAP)y[index]=DIM_MAP-1;
	if(y[index]<0)y[index]=0;
}
