__kernel void stepV(__global float *x, __global float *y, __global float *z, __global float *w,
		__global float *vx, __global float *vy, __global float *vz, __global float *vw,
		__global float *preDistances)
{
	__local float lx[NO_PARTICLES];
	__local float ly[NO_PARTICLES];
	__local float lz[NO_PARTICLES];
	__local float lw[NO_PARTICLES];

	int gid = get_global_id(0)*BATCH;
	float fx;
	float fy;
	float fz;
	float fw;
	int j;
	float dx;
	float dy;
	float dz;
	float dw;

	float lvx[BATCH];
	float lvy[BATCH];
	float lvz[BATCH];
	float lvw[BATCH];
	int f;
	float d;
	int batch;
	int i;

	for (batch=0;batch<BATCH;batch++)
	{
		lx[gid+batch]=x[gid+batch];
		ly[gid+batch]=y[gid+batch];
		lz[gid+batch]=z[gid+batch];
		lw[gid+batch]=w[gid+batch];
		lvx[batch]=vx[gid+batch];
		lvy[batch]=vy[gid+batch];
		lvz[batch]=vz[gid+batch];
		lvw[batch]=vw[gid+batch];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (f=0;f<1000;f++)
	{
		for (batch=0;batch<BATCH;batch++)
		{
			fx=0;
			fy=0;
			fz=0;
			fw=0;
			i=gid+batch;
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
			fx-=lvx[batch]*FRICTION;
			fy-=lvy[batch]*FRICTION;
			fz-=lvz[batch]*FRICTION;
			fw-=lvw[batch]*FRICTION;
			lvx[batch]+=fx*DT;
			lvy[batch]+=fy*DT;
			lvz[batch]+=fz*DT;
			lvw[batch]+=fw*DT;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		for (batch=0;batch<BATCH;batch++)
		{
			lx[gid+batch]+=lvx[batch]*DT;
			ly[gid+batch]+=lvy[batch]*DT;
			lz[gid+batch]+=lvz[batch]*DT;
			lw[gid+batch]+=lvw[batch]*DT;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	for (batch=0;batch<BATCH;batch++)
	{
		x[gid+batch]=lx[gid+batch];
		y[gid+batch]=ly[gid+batch];
		z[gid+batch]=lz[gid+batch];
		w[gid+batch]=lw[gid+batch];
		vx[gid+batch]=lvx[batch];
		vy[gid+batch]=lvy[batch];
		vz[gid+batch]=lvz[batch];
		vw[gid+batch]=lvw[batch];
	}
}
