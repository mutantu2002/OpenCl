__kernel void stepV(__global double *x, __global double *y, __global double *z, __global double *w,
		__global double *vx, __global double *vy, __global double *vz, __global double *vw,
		__global double *preDistances)
{
	__local double lx[NO_PARTICLES];
	__local double ly[NO_PARTICLES];
	__local double lz[NO_PARTICLES];
	__local double lw[NO_PARTICLES];

	int i = get_global_id(0);
	double fx=0;
	double fy=0;
	double fz=0;
	double fw=0;
	int j;
	double dx;
	double dy;
	double dz;
	double dw;

	double lvx;
	double lvy;
	double lvz;
	double lvw;
	int f;
	double d;

	lx[i]=x[i];
	ly[i]=y[i];
	lz[i]=z[i];
	lw[i]=w[i];
	lvx=vx[i];
	lvy=vy[i];
	lvz=vz[i];
	lvw=vw[i];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (f=0;f<150;f++)
	{
		fx=0;
		fy=0;
		fz=0;
		fw=0;
		for (j=0;j<NO_PARTICLES;j++){
			if (i==j)continue;
			dx=lx[i]-lx[j];
			dy=ly[i]-ly[j];
			dz=lz[i]-lz[j];
			dw=lw[i]-lw[j];
			d = sqrt(dx*dx+dy*dy+dz*dz+dw*dw);
			fx+=dx/d*(preDistances[i*NO_PARTICLES+j]-d)*K;
			fy+=dy/d*(preDistances[i*NO_PARTICLES+j]-d)*K;
			fz+=dz/d*(preDistances[i*NO_PARTICLES+j]-d)*K;
			fw+=dw/d*(preDistances[i*NO_PARTICLES+j]-d)*K;
		}
		fx-=lvx*FRICTION;
		fy-=lvy*FRICTION;
		fz-=lvz*FRICTION;
		fw-=lvw*FRICTION;
		lvx+=fx*DT;
		lvy+=fy*DT;
		lvz+=fz*DT;
		lvw+=fw*DT;

		barrier(CLK_LOCAL_MEM_FENCE);

		lx[i]+=lvx*DT;
		ly[i]+=lvy*DT;
		lz[i]+=lvz*DT;
		lw[i]+=lvw*DT;

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	x[i]=lx[i];
	y[i]=ly[i];
	z[i]=lz[i];
	w[i]=lw[i];
	vx[i]=lvx;
	vy[i]=lvy;
	vz[i]=lvz;
	vw[i]=lvw;
}
