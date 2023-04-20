#include "absl/base/config.h"
#undef ABSL_HAVE_STD_STRING_VIEW

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <list>
#include <functional>
#include "backend.hpp"
#include <chrono>
using namespace std::chrono;
using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("TmciRestore")
    .Attr("backend: string")
    .Attr("config: string")
    .Attr("model_ids: list(int)")
    .Attr("lids: list(int)")
    .Attr("lowners: list(int)")
    //.Attr("sizes: list(int)")
    .Input("tensors: T")
    .Attr("T: list(type)")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            return Status::OK();
    });

/**
 * @brief Generic restore operation for Tensorflow.
 * This operation takes a backend name, a configuration,
 * and a list of tensors. It instanciates the request
 * backend and calls its Load function.
 */
class TMCIRestoreOp : public OpKernel {

    public:

    explicit TMCIRestoreOp(OpKernelConstruction* context)
    : OpKernel(context) {
        uint64_t ts_start = duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
        std::string backend;
        std::string config;
        OP_REQUIRES_OK(context, context->GetAttr("backend", &backend));
        OP_REQUIRES_OK(context, context->GetAttr("config", &config));
        OP_REQUIRES_OK(context, context->GetAttr("model_ids", &model_ids));
        OP_REQUIRES_OK(context, context->GetAttr("lids", &lids));
        
        OP_REQUIRES_OK(context, context->GetAttr("lowners", &lowners));
        //OP_REQUIRES_OK(context, context->GetAttr("sizes", &sizes));
        m_backend = tmci::Backend::Create(backend.c_str(), config.c_str());
        uint64_t ts_end = duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
    }

    void Compute(OpKernelContext* context) override {
        unsigned n = context->num_inputs();
        std::vector< std::reference_wrapper<const tensorflow::Tensor>> tensors;
        tensors.reserve(n);
        for(unsigned i=0; i < n; i++) {
            tensors.push_back(std::cref(context->input(i)));
        }
        int status = m_backend->Load(tensors, model_ids, lids, lowners);
        OP_REQUIRES(context, status == 0, errors::Internal(status));
    }

    private:

    std::shared_ptr<tmci::Backend> m_backend;
	std::vector<int32_t>lids;
	std::vector<int32_t>lowners;
	//std::vector<int32_t>sizes;
	std::vector<int32_t>model_ids;
};

REGISTER_KERNEL_BUILDER(Name("TmciRestore").Device(DEVICE_CPU), TMCIRestoreOp);
REGISTER_KERNEL_BUILDER(Name("TmciRestore").Device(DEVICE_GPU), TMCIRestoreOp);
