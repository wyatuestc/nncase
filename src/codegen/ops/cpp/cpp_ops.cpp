/* Copyright 2019 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <codegen/codegen.h>
#include <fmt/format.h>
#include <ir/op_utils.h>
#include <ir/ops/binary.h>
#include <ir/ops/concat.h>
#include <ir/ops/constant.h>
#include <ir/ops/conv2d.h>
#include <ir/ops/dequantize.h>
#include <ir/ops/fake_dequantize.h>
#include <ir/ops/fake_quantize.h>
#include <ir/ops/matmul.h>
#include <ir/ops/pad.h>
#include <ir/ops/quantize.h>
#include <ir/ops/reduce.h>
#include <ir/ops/reduce_window2d.h>
#include <ir/ops/reshape.h>
#include <ir/ops/resize_image.h>
#include <ir/ops/strided_slice.h>
#include <ir/ops/transpose.h>
#include <ir/ops/unary.h>
#include <runtime/neutral/neutral_ops_body.h>
#include <sstream>

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::runtime;
using namespace nncase::ir;
using namespace nncase::runtime::neutral;

namespace
{
std::string_view get_type_name(datatype_t type)
{
    switch (type)
    {
    case dt_float32:
        return "float";
    case dt_uint8:
        return "uint8_t";
    default:
        throw std::invalid_argument("Unsupported datatype");
    }
}

template <runtime::runtime_opcode Op, class T>
struct cpp_body_impl : public T, public node_body
{
    std::stringstream str;

    runtime::runtime_opcode opcode() const noexcept override
    {
        return Op;
    }

    void serialize(runtime::binary_writer &writer) override
    {
        auto text = str.str();
        writer.write_string(text);
    }

    void begin(node &node)
    {
        str << "kernel_call_result " << node.name() << "(interpreter &interp)\n{\n";
    }

    void end()
    {
        str << "}\n";
    }

    cpp_body_impl &ident(size_t n = 1)
    {
        for (size_t i = 0; i < n; i++)
            str << "    ";
        return *this;
    }

    void allocation(codegen_context &context, ir::input_connector &conn, std::string_view name)
    {
        if (conn.connection()->owner().runtime_opcode() == op_constant)
        {
            auto &con = static_cast<ir::constant &>(conn.connection()->owner());
            str << fmt::format("auto {} = c_{};", name, con.name());
        }
        else
        {
            auto alloc = context.get_allocation(conn);
            assert(alloc.memory_type == mem_main);
            str << fmt::format("auto {} = reinterpret_cast<const {} *>(interp.main_memory() + {});", name, get_type_name(conn.type()), alloc.start) << std::endl;
        }
    }

    void allocation(codegen_context &context, ir::output_connector &conn, std::string_view name)
    {
        auto alloc = context.get_allocation(conn);
        assert(alloc.memory_type == mem_main);
        str << fmt::format("auto {} = reinterpret_cast<{} *>(interp.main_memory() + {});", name, get_type_name(conn.type()), alloc.start) << std::endl;
    }

    template <class T>
    void constant(const T *begin, const T *end, std::string_view name, std::string_view type)
    {
        str << fmt::format("static const {} {}[] = {{ ", type, name);
        for (auto it = begin; it < end; it++)
            str << *it << ", ";
        str << " };" << std::endl;
    }
};
}

namespace nncase
{
namespace codegen
{
    void register_cpp_emitters()
    {
        disable_emitter(op_input_node);
        disable_emitter(op_output_node);
        disable_emitter(op_ignore_node);
        disable_emitter(op_constant);

        register_emitter(op_binary, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<binary &>(node);
            auto body = std::make_unique<node_body_impl<rop_binary, binary_options>>();

            body->input_a = context.get_allocation(rnode.input_a());
            body->input_b = context.get_allocation(rnode.input_b());
            body->output = context.get_allocation(rnode.output());
            body->binary_op = rnode.binary_op();
            body->in_a_shape = to(rnode.input_a().shape());
            body->in_b_shape = to(rnode.input_b().shape());
            body->out_shape = to(rnode.output().shape());
            body->fused_activation = rnode.fused_activation();

            return body;
        });

        register_emitter(op_concat, [](node &node, codegen_context &context) {
            struct concat_options_body : public node_body_impl<rop_concat, concat_options>
            {
                std::vector<memory_range> inputs_holder;
            };

            auto &rnode = static_cast<concat &>(node);
            auto body = std::make_unique<concat_options_body>();

            for (auto &&in : rnode.inputs())
                body->inputs_holder.emplace_back(context.get_allocation(in));

            auto elem_size = (uint32_t)runtime::get_bytes(rnode.output().type());
            uint64_t inner_size, outer_size;
            get_concat_params(rnode.output().shape(), elem_size, rnode.axis(), inner_size, outer_size);

            body->output = context.get_allocation(rnode.output());
            body->inner_size = inner_size;
            body->outer_size = outer_size;
            body->inputs_count = (uint32_t)body->inputs_holder.size();
            body->inputs = body->inputs_holder;
            body->dims = rnode.concat_dims();

            return body;
        });

        register_emitter(op_conv2d, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<conv2d &>(node);
            auto body = std::make_unique<cpp_body_impl<rop_conv2d, conv2d_options>>();
            auto in_shape = rnode.input().shape();
            auto out_shape = rnode.output().shape();
            auto w_shape = rnode.weights().shape();
            const auto g_ic = in_shape[1] / rnode.groups();
            const auto g_oc = rnode.output_channels() / rnode.groups();

            body->begin(node);

            body->ident().constant(rnode.weights().begin(), rnode.weights().end(), "weights", "float");
            body->ident().constant(rnode.bias().begin(), rnode.bias().end(), "bias", "float");

            body->str << "\n";
            body->ident().allocation(context, rnode.input(), "input");
            body->ident().allocation(context, rnode.output(), "output");

            body->str << fmt::format(R"(
    for (int32_t batch = 0; batch < {0}; batch++)
    {{
        const float *in_batch_p = input + (size_t)batch * {1} * {2} * {3};
    
        for (int32_t og = 0; og < {4}; og++)
        {{
            const float *in_group_p = in_batch_p + (size_t)og * {5} * {2} * {3};
            const float *w_group_p = weights + (size_t)og * {6} * {5} * {7} * {8};
    
            for (int32_t oc = 0; oc < {6}; oc++)
            {{
                const float *w_oc_p = w_group_p + (size_t)oc * {5} * {7} * {8};
    
                for (int32_t oy = 0; oy < {9}; oy++)
                {{
                    for (int32_t ox = 0; ox < {10}; ox++)
                    {{
                        const int32_t in_y_origin = (oy * {11}) - {13};
                        const int32_t in_x_origin = (ox * {12}) - {14};
                        const int32_t filter_y_start = std::max(0, (-in_y_origin + {15} - 1) / {15});
                        const int32_t filter_y_end = std::min({7}, ({2} - in_y_origin + {15} - 1) / {15});
                        const int32_t filter_x_start = std::max(0, (-in_x_origin + {16} - 1) / {16});
                        const int32_t filter_x_end = std::min({8}, ({3} - in_x_origin + {16} - 1) / {16});
                        float value = bias[og * {6} + oc];
    
                        for (int32_t ic = 0; ic < {5}; ic++)
                        {{
                            const float *in_c_p = in_group_p + (size_t)ic * {2} * {3};
                            const float *w_ic_p = w_oc_p + (size_t)ic * {7} * {8};
    
                            for (int32_t ky = filter_y_start; ky < filter_y_end; ky++)
                            {{
                                for (int32_t kx = filter_x_start; kx < filter_x_end; kx++)
                                {{
                                    const int32_t in_y = in_y_origin + {15} * ky;
                                    const int32_t in_x = in_x_origin + {16} * kx;
    
                                    const float in_v = in_c_p[in_y * {3} + in_x];
                                    const float w = w_ic_p[ky * {8} + kx];
    
                                    value += in_v * w;
                                }}
                            }}
                        }}
    
                        *output++ = details::apply_activation<float>(value, {17}, {18});
                    }}
                }}
            }}
        }}
    }}

    return kcr_done;
)",
                in_shape[0], in_shape[1], in_shape[2], in_shape[3], rnode.groups(), g_ic, g_oc, w_shape[2], w_shape[3], out_shape[2], out_shape[3],
                rnode.stride_h(), rnode.stride_w(), rnode.padding_h().before, rnode.padding_w().before, rnode.dilation_h(), rnode.dilation_w(),
                rnode.fused_activation().min, rnode.fused_activation().max);

            body->end();

            return body;
        });

        register_emitter(op_dequantize, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<dequantize &>(node);
            auto body = std::make_unique<node_body_impl<rop_dequantize, dequantize_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->quant_param = rnode.quant_param();

            return body;
        });

        register_emitter(op_matmul, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<matmul &>(node);
            auto body = std::make_unique<node_body_impl<rop_matmul, matmul_options>>();

            body->input_a = context.get_allocation(rnode.input_a());
            body->input_b = context.get_allocation(rnode.input_b());
            body->output = context.get_allocation(rnode.output());
            body->a_rows = rnode.input_a().shape()[0];
            body->a_cols = rnode.input_a().shape()[1];
            body->b_cols = rnode.input_b().shape()[1];
            body->fused_activation = rnode.fused_activation();
            body->bias = rnode.bias();

            return body;
        });

        register_emitter(op_pad, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<pad &>(node);
            auto body = std::make_unique<node_body_impl<rop_pad, pad_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->in_shape = to(rnode.input().shape());
            body->paddings = to(rnode.paddings());

            return body;
        });

        register_emitter(op_quantize, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<quantize &>(node);
            auto body = std::make_unique<node_body_impl<rop_quantize, quantize_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->quant_param = rnode.quant_param();

            return body;
        });

        register_emitter(op_reduce, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<reduce &>(node);
            auto body = std::make_unique<node_body_impl<rop_reduce, reduce_options>>();

            auto reduced_shape = get_reduced_shape(rnode.input().shape(), rnode.axis(), true);

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->reduce_op = rnode.reduce_op();
            body->in_shape = to(rnode.input().shape());
            body->out_shape = to(reduced_shape);
            body->init_value = rnode.init_value();

            return body;
        });

        register_emitter(op_reduce_window2d, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<reduce_window2d &>(node);
            auto body = std::make_unique<node_body_impl<rop_reduce_window2d, reduce_window2d_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->reduce_op = rnode.reduce_op();
            body->in_shape = to(rnode.input().shape());
            body->padding_h = rnode.padding_h();
            body->padding_w = rnode.padding_w();
            body->filter_h = rnode.filter_h();
            body->filter_w = rnode.filter_w();
            body->stride_h = rnode.stride_h();
            body->stride_w = rnode.stride_w();
            body->dilation_h = rnode.dilation_h();
            body->dilation_w = rnode.dilation_w();
            body->init_value = rnode.init_value();
            body->fused_activation = rnode.fused_activation();

            return body;
        });

        register_emitter(op_reshape, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<reshape &>(node);
            auto body = std::make_unique<node_body_impl<rop_memory_copy, memory_copy_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());

            return body;
        });

        register_emitter(op_resize_image, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<resize_image &>(node);
            auto body = std::make_unique<node_body_impl<rop_resize_image, resize_image_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->in_shape = to(rnode.input().shape());
            body->out_h = rnode.new_size()[0];
            body->out_w = rnode.new_size()[1];
            body->mode = rnode.mode();
            body->align_corners = rnode.align_corners();

            return body;
        });

        register_emitter(op_strided_slice, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<strided_slice &>(node);
            auto body = std::make_unique<node_body_impl<rop_strided_slice, strided_slice_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->in_shape = to(rnode.input().shape());
            body->begin = to(rnode.begin());
            body->end = to(rnode.begin());
            body->strides = to(rnode.begin());
            body->begin_mask = rnode.begin_mask();
            body->end_mask = rnode.end_mask();
            body->ellipsis_mask = rnode.ellipsis_mask();
            body->new_axis_mask = rnode.new_axis_mask();
            body->shrink_axis_mask = rnode.shrink_axis_mask();

            return body;
        });

        register_emitter(op_transpose, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<transpose &>(node);
            auto body = std::make_unique<node_body_impl<rop_transpose, transpose_options>>();

            runtime_shape_t in_shape, perm;
            extend_transpose_shape(rnode.input().shape(), rnode.perm(), in_shape, perm);

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->in_shape = in_shape;
            body->perm = perm;

            return body;
        });

        register_emitter(op_unary, [](node &node, codegen_context &context) {
            auto &rnode = static_cast<unary &>(node);
            auto body = std::make_unique<node_body_impl<rop_unary, unary_options>>();

            body->input = context.get_allocation(rnode.input());
            body->output = context.get_allocation(rnode.output());
            body->unary_op = rnode.unary_op();

            return body;
        });
    }
}
}
