package home.mutant.opencl.model;

import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_READ_WRITE;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clEnqueueWriteBuffer;
import static org.jocl.CL.clReleaseMemObject;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;

public class MemoryShort {
	Program program;
	cl_mem clMemObject;
	short[] src;
	
	public MemoryShort(Program program) {
		super();
		this.program = program;
	}

	public void addReadOnly(short[] src){
		this.src = src;
		clMemObject = clCreateBuffer(program.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_short * src.length, Pointer.to(src), null);
		copyHtoD();
	}
	
	public void addReadWrite(short[] src){
		this.src = src;
		clMemObject = clCreateBuffer(program.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_short * src.length, Pointer.to(src), null);
		copyHtoD();
	}
	public void copyDtoH()
	{
        clEnqueueReadBuffer(program.getCommandQueue(), clMemObject, CL_TRUE, 0,  src.length * Sizeof.cl_short, Pointer.to(src), 0, null, null);
	}
	public int copyHtoD()
	{
        return clEnqueueWriteBuffer(program.getCommandQueue(), clMemObject, CL_TRUE, 0,  src.length * Sizeof.cl_short, Pointer.to(src), 0, null, null);
	}
	public short[] getSrc() {
		return src;
	}
	public cl_mem gemMemObject() {
		return clMemObject;
	}
	public void release()
	{
		clReleaseMemObject(clMemObject);
	}
}
