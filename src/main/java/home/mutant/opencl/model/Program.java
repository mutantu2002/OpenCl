package home.mutant.opencl.model;

import static org.jocl.CL.*;

import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.Scanner;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;


public class Program 
{
	cl_command_queue clCommandQueue;
	cl_program clProgram;
	cl_context clContext;
	cl_device_id device;
	
	public Program(String source)
	{
		this(source,null);
	}
	public Program(String source, Map<String, Object> params)
	{
        final int platformIndex = 0;
        final long deviceType = CL_DEVICE_TYPE_GPU;
        final int deviceIndex = 0;

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);

        // Obtain the number of platforms
        int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
        
        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];
        
        // Obtain a device ID 
        cl_device_id devices[] = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        device = devices[deviceIndex];

        long[] size=new long[1];
        clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, Sizeof.cl_ulong, Pointer.to(size), null);
        
        clContext = clCreateContext(contextProperties, 1, new cl_device_id[]{device}, null, null, null);
        clCommandQueue = clCreateCommandQueue(clContext, device, 0, null);
        clProgram = clCreateProgramWithSource(clContext, 1, new String[]{ source }, null, null);
        String options=null;
        if(params!=null){
        	options="";
        	for(String param:params.keySet()){
        		options+="-D"+param+"="+params.get(param).toString()+" ";
        	}
        }
        clBuildProgram(clProgram, 0, null, options, null, null);
	}
	
	public void release()
	{
        clReleaseProgram(clProgram);
        clReleaseCommandQueue(clCommandQueue);
        clReleaseContext(clContext);
	}
	public int finish()
	{
		return clFinish(clCommandQueue);
	}

	public cl_context getContext() {
		return clContext;
	}

	public cl_program getProgram() {
		return clProgram;
	}

	public cl_command_queue getCommandQueue() {
		return clCommandQueue;
	}
	public static String readResource(String resource){
    	InputStream resourceAsStream = Program.class.getResourceAsStream(resource);
		Scanner scanner  = new Scanner(resourceAsStream, "UTF-8");
		String text = scanner.useDelimiter("\\A").next();
		scanner.close();
		try {
			resourceAsStream.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return text;
	}
	public cl_device_id getDevice() {
		return device;
	}
	public void setDevice(cl_device_id device) {
		this.device = device;
	}
}
