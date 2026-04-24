#define CL_TARGET_OPENCL_VERSION 300
#include <vector>
#include <CL/cl.h>
#include "gelu_ocl.h"


const char* kernel_source = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        float argument = 1.595769f * (x + 0.044715f * x * x * x);
        output[i] = x - x / (exp(argument) + 1.0f);
    }
}
)";



std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {

    cl_uint num_platforms;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    static cl_device_id device;
    static cl_context context;
    static cl_command_queue queue;
    static cl_program program;
    static cl_kernel kernel;
    static bool is_init = false;

    if (!is_init) {
        cl_platform_id pid = platforms[platform];
        clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);

        cl_queue_properties props[] = {0}; 
        queue = clCreateCommandQueueWithProperties(context, device, props, nullptr);

        program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, nullptr);
        clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        kernel = clCreateKernel(program, "gelu_kernel", nullptr);

        is_init = true;
    }


    size_t n = input.size();
    size_t bytes = n * sizeof(float);
    cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      bytes, const_cast<float*>(input.data()), nullptr);
    cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);

    int n_for_kernel = static_cast<int>(n);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &gpu_output);
    clSetKernelArg(kernel, 2, sizeof(int), &n_for_kernel);

    size_t work_size = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &work_size, nullptr, 0, nullptr, nullptr);


    std::vector<float> result(n);

    clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, bytes, result.data(), 0, nullptr, nullptr);

    clReleaseMemObject(gpu_input);
    clReleaseMemObject(gpu_output);

    return result;
}