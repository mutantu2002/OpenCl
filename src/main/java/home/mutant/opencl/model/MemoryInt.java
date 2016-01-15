package home.mutant.opencl.model;

import static org.jocl.CL.*;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;

public class MemoryInt {
	Program program;
	cl_mem clMemObject;
	int[] src;
	
	public MemoryInt(Program program) {
		super();
		this.program = program;
	}

	public void addReadOnly(int[] src){
		this.src = src;
		clMemObject = clCreateBuffer(program.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * src.length, Pointer.to(src), null);
		copyHtoD();
	}
	
	public void addReadWrite(int[] src){
		this.src = src;
		clMemObject = clCreateBuffer(program.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * src.length, Pointer.to(src), null);
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
	public int[] getSrc() {
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
