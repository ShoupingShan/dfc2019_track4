#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
using namespace tensorflow;

REGISTER_OP("KernelCorrelation")
    .Attr("sigma: float")
    .Input("points: float32")
    .Input("kernel: float32")
    .Output("features: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    //points: (batch_size, n, neighbor,3) float32 array, points to sample from
    //kernel: (l, m, 3) int32 array, indices to points
    //output(BS, l, n)
        ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * ndataset * channels
        c->WithRank(c->input(0), 4, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoints * nsample
        c->WithRank(c->input(1), 3, &dims2);
        // batch_size * npoints * nsample * channels
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims2, 0), c->Dim(dims1, 1)});
        c->set_output(0, output);
        return Status::OK();
    });


REGISTER_OP("KernelCorrelationGrad")
    .Attr("sigma: float")
    .Input("points: float32")
    .Input("kernel: float32")
    .Input("grad_features: float32")
    .Output("grad_kernel: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        return Status::OK();
    });

void kernel_correlation_kernel_wrapper(int b, int n, int npoints, int l, int m, float sigma,const float *points, const float *kernel,float *outputs);

class KernelCorrelationOp: public OpKernel{
    public:
        explicit KernelCorrelationOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("sigma", &sigma_));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==4, errors::InvalidArgument("KernelCorrelation expects (batch_size, num_centers, num_points, channel) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int n = points_tensor.shape().dim_size(1);
            int npoints = points_tensor.shape().dim_size(2);

            const Tensor& kernel_tensor=context->input(1);
            int kernel_l = kernel_tensor.shape().dim_size(0);
            int kernel_m = kernel_tensor.shape().dim_size(1);
            OP_REQUIRES(context, kernel_m < npoints, errors::InvalidArgument("npoints must smaller than kernel length m"));

            Tensor *features_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, kernel_l, n}, &features_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));

            auto kernel_flat = kernel_tensor.flat<float>();
            const float *kernel = &(kernel_flat(0));

            auto features_flat = features_tensor->flat<float>();
            float *features = &(features_flat(0));

            kernel_correlation_kernel_wrapper(b, n, npoints, kernel_l, kernel_m, sigma_, points, kernel, features);

        }

    private:
        float sigma_;
};

REGISTER_KERNEL_BUILDER(Name("KernelCorrelation").Device(DEVICE_GPU), KernelCorrelationOp);


void kernel_correlation_grad_kernel_wrapper(int b, int n, int npoints, int l, int m, float sigma, const float *grad_outputs, const float *points, const float *kernel,float *grad_inputs);

class KernelCorrelationGradOp: public OpKernel{
    public:
        explicit KernelCorrelationGradOp(OpKernelConstruction * context):OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("sigma", &sigma_));
        }

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==4, errors::InvalidArgument("KernelCorrelationGrad expects (batch_size, num_centers, num_points, channel) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int n = points_tensor.shape().dim_size(1);
            int npoints = points_tensor.shape().dim_size(2);

            const Tensor& kernel_tensor=context->input(1);
            int kernel_l = kernel_tensor.shape().dim_size(0);
            int kernel_m = kernel_tensor.shape().dim_size(1);

            const Tensor& grad_features_tensor=context->input(2);
            OP_REQUIRES(context,grad_features_tensor.dims()==3 && grad_features_tensor.shape().dim_size(0)==b && grad_features_tensor.shape().dim_size(1)==kernel_l && grad_features_tensor.shape().dim_size(2)==n, errors::InvalidArgument("KernelCorrelationGrad expects (batch_size, kernel_num, nsample) grad_features shape"));

            Tensor *grad_kernel_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{kernel_l, kernel_m, 3}, &grad_kernel_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));
            auto kernel_flat = kernel_tensor.flat<float>();
            const float *kernel = &(kernel_flat(0));
            auto grad_features_flat = grad_features_tensor.flat<float>();
            const float *grad_features = &(grad_features_flat(0));

            auto grad_kernel_flat = grad_kernel_tensor->flat<float>();
            float *grad_kernel = &(grad_kernel_flat(0));
            cudaMemset(grad_kernel, 0, sizeof(float)*kernel_l*kernel_m*3);
            kernel_correlation_grad_kernel_wrapper(b, n, npoints, kernel_l, kernel_m, sigma_, grad_features, points, kernel, grad_kernel);
        }
    private:
        float sigma_;
};
REGISTER_KERNEL_BUILDER(Name("KernelCorrelationGrad").Device(DEVICE_GPU), KernelCorrelationGradOp);



