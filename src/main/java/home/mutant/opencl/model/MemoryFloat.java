package home.mutant.opencl.model;

import static org.jocl.CL.*;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;

public class MemoryFloat {
	Program program;
	cl_mem clMemObject;
	float[] src;
	
	public MemoryFloat(Program program) {
		super();
		this.program = program;
	}

	public void addReadOnly(float[] src){
		this.src = src;
		clMemObject = clCreateBuffer(program.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * src.length, Pointer.to(src), null);
		copyHtoD();
	}
	
	public void addReadWrite(float[] src){
		this.src = src;
		clMemObject = clCreateBuffer(program.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * src.length, Pointer.to(src), null);
		copyHtoD();
	}
	public void copyDtoH()
	{
        clEnqueueReadBuffer(program.getCommandQueue(), clMemObject, CL_TRUE, 0,  src.length * Sizeof.cl_float, Pointer.to(src), 0, null, null);
	}
	public int copyHtoD()
	{
        return clEnqueueWriteBuffer(program.getCommandQueue(), clMemObject, CL_TRUE, 0,  src.length * Sizeof.cl_float, Pointer.to(src), 0, null, null);
	}
	public float[] getSrc() {
		return src;
	}
	public cl_mem gemMemObject() {
		// TODO Auto-generated method stub
		return clMemObject;
	}
	public void release()
	{
		clReleaseMemObject(clMemObject);
	}
}
