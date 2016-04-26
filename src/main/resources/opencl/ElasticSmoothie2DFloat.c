__kernel void stepV(__global float *x, __global float *y,
		__global float *vx, __global float *vy,
		__global float *preDistances)
{
	__local float lx[NO_PARTICLES];
	__local float ly[NO_PARTICLES];

	int i = get_global_id(0);
	float fx=0;
	float fy=0;

	int j;
	float dx;
	float dy;

	float lvx;
	float lvy;
	int f;
	float d;

	lx[i]=x[i];
	ly[i]=y[i];
	lvx=vx[i];
	lvy=vy[i];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (f=0;f<150;f++)
	{
		fx=0;
		fy=0;
		for (j=0;j<NO_PARTICLES;j++){
			if (i==j)continue;
			dx=lx[i]-lx[j];
			dy=ly[i]-ly[j];
			d = sqrt(dx*dx+dy*dy);
			fx+=dx/d*(preDistances[i*NO_PARTICLES+j]-d)*K;
			fy+=dy/d*(preDistances[i*NO_PARTICLES+j]-d)*K;
		}
		fx-=lvx*FRICTION;
		fy-=lvy*FRICTION;
		lvx+=fx*DT;
		lvy+=fy*DT;

		barrier(CLK_LOCAL_MEM_FENCE);

		lx[i]+=lvx*DT;
		ly[i]+=lvy*DT;

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	x[i]=lx[i];
	y[i]=ly[i];
	vx[i]=lvx;
	vy[i]=lvy;
}
