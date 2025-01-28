#include <ATen/ATen.h>

// Function declarations
int corr_cuda_forward(at::Tensor input1,
                      at::Tensor input2,
                      at::Tensor rbot1,
                      at::Tensor rbot2,
                      at::Tensor output,
                      int pad_size,
                      int kernel_size,
                      int max_displacement,
                      int stride1,
                      int stride2,
                      int corr_type_multiply);

int corr_cuda_backward(at::Tensor input1,
                       at::Tensor input2,
                       at::Tensor rbot1,
                       at::Tensor rbot2,
                       at::Tensor grad_output,
                       at::Tensor grad_input1,
                       at::Tensor grad_input2,
                       int pad_size,
                       int kernel_size,
                       int max_displacement,
                       int stride1,
                       int stride2,
                       int corr_type_multiply);