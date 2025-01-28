#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "corr1d_cuda_kernel.h"

// == Forward
int corr1d_cuda_forward(at::Tensor input1,
                        at::Tensor input2,
                        at::Tensor rbot1,
                        at::Tensor rbot2,
                        at::Tensor output,
                        int pad_size,
                        int kernel_size,
                        int max_displacement,
                        int stride1,
                        int stride2,
                        int corr_type_multiply)
{
    // TODO: Shape checks

    int batchSize = input1.size(0);
    long nInputPlane = input1.size(1);
    long nInputRows = input1.size(2);
    long nInputCols = input1.size(3);
    long inputWidthHeight = nInputRows * nInputCols;

    long kernel_radius_ = (kernel_size - 1) / 2;
    long border_size_ = max_displacement + kernel_radius_; // size of unreachable border region (on each side)

    long paddedbottomheight = nInputRows;
    long paddedbottomwidth = nInputCols + 2 * pad_size;

    long nOutputCols = ceil((float)(paddedbottomwidth - border_size_ * 2) / (float)stride1);
    long nOutputRows = ceil((float)(paddedbottomheight - kernel_radius_ * 2) / (float)stride1);

    long neighborhood_grid_radius_ = max_displacement / stride2;
    long neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;
    int x_shift = -neighborhood_grid_radius_;

    int nOutputPlane = neighborhood_grid_width_; // Number of output channels for 1D X-correlation

    // Resize and initialize output tensors
    output = at::zeros({batchSize, nOutputPlane, nOutputRows, nOutputCols}, input1.options());
    rbot1 = at::zeros({batchSize, nInputPlane, paddedbottomheight, paddedbottomwidth}, input1.options());
    rbot2 = at::zeros({batchSize, nInputPlane, paddedbottomheight, paddedbottomwidth}, input1.options());

    float *input1_data = input1.data_ptr<float>();
    float *input2_data = input2.data_ptr<float>();
    float *output_data = output.data_ptr<float>();
    float *rbot1_data = rbot1.data_ptr<float>();
    float *rbot2_data = rbot2.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int pwidthheight = paddedbottomwidth * paddedbottomheight;

    blob_rearrange_ongpu_1d(input1_data, rbot1_data, batchSize, nInputPlane, nInputCols, nInputRows, inputWidthHeight, pad_size, pwidthheight, stream);
    blob_rearrange_ongpu_1d(input2_data, rbot2_data, batchSize, nInputPlane, nInputCols, nInputRows, inputWidthHeight, pad_size, pwidthheight, stream);

    CorrelateData_ongpu_1d(rbot1_data, rbot2_data, output_data, batchSize, nOutputCols, nOutputRows, nOutputPlane, max_displacement, x_shift, neighborhood_grid_width_, kernel_radius_, kernel_size, stride1, stride2, paddedbottomwidth, paddedbottomheight, nInputPlane, corr_type_multiply, stream);

    return 1;
}

int corr1d_cuda_backward(at::Tensor input1,
                         at::Tensor input2,
                         at::Tensor rbot1,
                         at::Tensor rbot2,
                         at::Tensor gradOutput,
                         at::Tensor gradInput1,
                         at::Tensor gradInput2,
                         int pad_size,
                         int kernel_size,
                         int max_displacement,
                         int stride1,
                         int stride2,
                         int corr_type_multiply)
{
    float *input1_data = input1.data_ptr<float>();
    float *input2_data = input2.data_ptr<float>();

    long nInputCols = input1.size(3);
    long nInputRows = input1.size(2);
    long nInputPlane = input1.size(1);
    long batchSize = input1.size(0);

    float *gradOutput_data = gradOutput.data_ptr<float>();
    float *gradInput1_data = gradInput1.data_ptr<float>();
    float *gradInput2_data = gradInput2.data_ptr<float>();

    long inputWidthHeight = nInputRows * nInputCols;

    long kernel_radius_ = (kernel_size - 1) / 2;
    long border_size_ = max_displacement + kernel_radius_; // size of unreachable border region (on each side)

    long paddedbottomheight = nInputRows;
    long paddedbottomwidth = nInputCols + 2 * pad_size;

    long nOutputCols = ceil((float)(paddedbottomwidth - border_size_ * 2) / (float)stride1);
    long nOutputRows = ceil((float)(paddedbottomheight - kernel_radius_ * 2) / (float)stride1);

    long neighborhood_grid_radius_ = max_displacement / stride2;
    long neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;
    int x_shift = -neighborhood_grid_radius_;

    int nOutputPlane = neighborhood_grid_width_; // Same for 1D X-correlation

    rbot1 = at::zeros({batchSize, nInputPlane, paddedbottomheight, paddedbottomwidth}, input1.options());
    rbot2 = at::zeros({batchSize, nInputPlane, paddedbottomheight, paddedbottomwidth}, input1.options());

    float *rbot1_data = rbot1.data_ptr<float>();
    float *rbot2_data = rbot2.data_ptr<float>();

    int pwidthheight = paddedbottomwidth * paddedbottomheight;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    blob_rearrange_ongpu_1d(input1_data, rbot1_data, batchSize, nInputPlane, nInputCols, nInputRows, inputWidthHeight, pad_size, pwidthheight, stream);
    blob_rearrange_ongpu_1d(input2_data, rbot2_data, batchSize, nInputPlane, nInputCols, nInputRows, inputWidthHeight, pad_size, pwidthheight, stream);

    CorrelateDataBackward_ongpu_1d(rbot1_data, rbot2_data, gradOutput_data, gradInput1_data, gradInput2_data, batchSize, nOutputCols, nOutputRows, nOutputPlane, max_displacement, x_shift, neighborhood_grid_width_, kernel_radius_, stride1, stride2, nInputCols, nInputRows, paddedbottomwidth, paddedbottomheight, nInputPlane, pad_size, corr_type_multiply, stream);

    return 1;
}