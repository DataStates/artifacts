#include "absl/base/config.h"
#undef ABSL_HAVE_STD_STRING_VIEW

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <iostream>
#include <functional>
#include<list>
#include "backend.hpp"

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("TmciCheckpoint")
.Attr("backend: string")
.Attr("config: string")
.Attr("model_ids: list(int)")
.Attr("lids: list(int)")
.Input("tensors: T")
    .Attr("T: list(type)")
.SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            return Status::OK();
            });

/**
 * @brief Generic checkpoint operation for Tensorflow.
 * This operation takes a backend name, a configuration,
 * and a list of tensors. It instanciates the request
 * backend and calls its Save function.
 */
class TMCICheckpointOp : public OpKernel {

    public:

        explicit TMCICheckpointOp(OpKernelConstruction* context)
            : OpKernel(context) {
                std::string backend;
                std::string config;
                OP_REQUIRES_OK(context, context->GetAttr("backend", &backend));
                OP_REQUIRES_OK(context, context->GetAttr("config", &config));
                OP_REQUIRES_OK(context, context->GetAttr("model_ids", &model_ids));
                OP_REQUIRES_OK(context, context->GetAttr("lids", &lids));
                m_backend = tmci::Backend::Create(backend.c_str(), config.c_str());
            }

        void Compute(OpKernelContext* context) override {
            unsigned n = context->num_inputs();
            std::vector<std::reference_wrapper<const tensorflow::Tensor>> tensors;
            tensors.reserve(n);
            for(unsigned i=0; i < n; i++) {
                tensors.push_back(std::cref(context->input(i)));
            }

            int status = m_backend->Save(tensors, model_ids, lids );

            OP_REQUIRES(context, status == 0, errors::Internal(status));
        }

    private:
        std::vector<int32_t> model_ids;
        std::vector<int32_t> lids;
        std::shared_ptr<tmci::Backend> m_backend;
};

REGISTER_KERNEL_BUILDER(Name("TmciCheckpoint").Device(DEVICE_CPU), TMCICheckpointOp);
REGISTER_KERNEL_BUILDER(Name("TmciCheckpoint").Device(DEVICE_GPU), TMCICheckpointOp);

