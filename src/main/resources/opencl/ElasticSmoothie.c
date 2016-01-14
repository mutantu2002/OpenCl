__kernel void stepV(__global double *x, __global double *y,__global double *vx, __global double *vy, __global double *preDistances)
{
	__local double lx[NO_PARTICLES];
	__local double ly[NO_PARTICLES];

	int i = get_global_id(0);
	double fx=0;
	double fy=0;
	int j;
	double dx;
	double dy;
	double lvx;
	double lvy;
	int f;
	double d;

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
