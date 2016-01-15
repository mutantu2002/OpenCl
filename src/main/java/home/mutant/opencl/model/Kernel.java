package home.mutant.opencl.model;

import static org.jocl.CL.*;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_kernel;

public class Kernel {
	Program program;
	cl_kernel clKernel;
	
	public Kernel(Program program, String functionName) {
		super();
		this.program = program;
		clKernel = clCreateKernel(program.getProgram(), functionName, null);
//        long[] size=new long[1];
//        clGetKernelWorkGroupInfo(clKernel,program.getDevice(), CL_KERNEL_WORK_GROUP_SIZE, Sizeof.cl_ulong, Pointer.to(size), null);
//        System.out.println(size[0]);
	}
	
	public void setArgument(MemoryDouble memory, int index){
		clSetKernelArg(clKernel, index, Sizeof.cl_mem, Pointer.to(memory.gemMemObject()));
	}
	
	public void setArgument(MemoryFloat memory, int index){
		clSetKernelArg(clKernel, index, Sizeof.cl_mem, Pointer.to(memory.gemMemObject()));
	}
	
	public void setArgument(MemoryInt memory, int index){
		clSetKernelArg(clKernel, index, Sizeof.cl_mem, Pointer.to(memory.gemMemObject()));
	}	
	public void setArgument(int value, int index){
		clSetKernelArg(clKernel, index, Sizeof.cl_int, Pointer.to(new int[]{ value }));
	}
	
	public void set2Argument(int value1,int value2, int index){
		clSetKernelArg(clKernel, index, Sizeof.cl_int2, Pointer.to(new int[]{ value1,value2 }));
	}
	
	public void setArguments(MemoryDouble ... memories){
		for (int i = 0; i < memories.length; i++) {
			clSetKernelArg(clKernel, i, Sizeof.cl_mem, Pointer.to(memories[i].gemMemObject()));
		}
	}
	public void setArguments(MemoryFloat ... memories){
		for (int i = 0; i < memories.length; i++) {
			clSetKernelArg(clKernel, i, Sizeof.cl_mem, Pointer.to(memories[i].gemMemObject()));
		}
	}
	public int run(long globalworkSize, long localWorksize)
	{
        return clEnqueueNDRangeKernel(program.getCommandQueue(), clKernel, 1, null, new long[]{globalworkSize}, new long[]{localWorksize}, 0, null, null);
	}
	public void release()
	{
		clReleaseKernel(clKernel);
	}
}
