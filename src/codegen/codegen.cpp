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
#include <fstream>
#include <ir/op_utils.h>
#include <ir/ops/constant.h>
#include <runtime/binary_writer.h>
#include <runtime/model.h>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::ir;
using namespace nncase::scheduler;
using namespace nncase::runtime;

namespace
{
std::unordered_map<node_opcode, emitter_t> g_emitters;
std::unordered_set<node_opcode> g_disabled_emitters;

std::unique_ptr<node_body> call_emitter(node &node, codegen_context &context)
{
    auto opcode = node.runtime_opcode();
    auto it = g_emitters.find(opcode);
    if (it == g_emitters.end())
    {
        if (g_disabled_emitters.find(opcode) == g_disabled_emitters.end())
            throw std::runtime_error(std::string("Emitter for ") + node_opcode_names(opcode).data() + " is not found");
    }
    else
    {
        return it->second(node, context);
    }

    return nullptr;
}

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
}

void nncase::codegen::register_emitter(node_opcode opcode, emitter_t emitter)
{
    g_emitters.emplace(opcode, std::move(emitter));
}

void nncase::codegen::disable_emitter(ir::node_opcode opcode)
{
    g_disabled_emitters.emplace(opcode);
}

codegen_context::codegen_context(const std::filesystem::path &output_path, const std::unordered_map<memory_type_t, memory_allocator *> &allocators, const std::unordered_map<ir::output_connector *, memory_allocation> &allocations)
    : output_path_(output_path), allocators_(allocators), allocations_(allocations)
{
}

memory_range codegen_context::get_allocation(output_connector &conn) const
{
    auto &alloc = allocations_.at(&conn);
    return { alloc.type, conn.type(), (uint32_t)alloc.start, (uint32_t)alloc.size };
}

void nncase::codegen::gen_kmodel(codegen_context &context, xtl::span<ir::node *> compute_sequence)
{
    std::vector<ir::node *> runtime_nodes;
    std::vector<memory_range> inputs;
    std::vector<runtime_shape_t> input_shapes;
    std::vector<memory_range> outputs;
    std::vector<ir::node *> constants;

    for (auto &&node : compute_sequence)
    {
        if (g_disabled_emitters.find(node->runtime_opcode()) == g_disabled_emitters.end())
            runtime_nodes.emplace_back(node);

        switch (node->runtime_opcode())
        {
        case op_input_node:
            inputs.emplace_back(context.get_allocation(node->output_at(0)));
            input_shapes.emplace_back(ir::to(node->output_at(0).shape()));
            break;
        case op_output_node:
            outputs.emplace_back(context.get_allocation(*node->input_at(0).connection()));
            break;
        case op_constant:
            constants.emplace_back(node);
            break;
        }
    }

    std::ofstream outfile(context.output_path(), std::ios::binary | std::ios::out);
    if (outfile.bad())
        throw std::runtime_error("Cannot open file for output: " + context.output_path().string());

    binary_writer writer(outfile);
    // model header
    model_header model_header;
    model_header.identifier = MODEL_IDENTIFIER;
    model_header.version = MODEL_VERSION;
    model_header.flags = 0;
    model_header.target = MODEL_TARGET_K210;
    model_header.constants = context.constant_usage();
    model_header.main_mem = context.memory_usage();
    model_header.nodes = runtime_nodes.size();
    model_header.inputs = inputs.size();
    model_header.outputs = outputs.size();

    writer.write(model_header);

    // inputs
    writer.write_array<memory_range>(inputs);
    // input shapes
    writer.write_array<runtime_shape_t>(input_shapes);
    // outputs
    writer.write_array<memory_range>(outputs);

    // constants
    auto const_mem = std::make_unique<uint8_t[]>(context.constant_usage());
    for (auto &node : constants)
    {
        auto &con = static_cast<constant &>(*node);
        auto alloc = context.get_allocation(con.output());
        auto start = const_mem.get() + alloc.start;
        std::copy(con.data().begin(), con.data().end(), start);
    }

    writer.write_array(xtl::span<const uint8_t> { const_mem.get(), context.constant_usage() });

    // Keep node headers
    std::vector<node_header> node_headers;
    auto node_headers_pos = writer.position();
    std::streamoff node_header_bytes = sizeof(node_header) * runtime_nodes.size();

    writer.position(node_headers_pos + node_header_bytes);

    // write body
    for (auto &&node : runtime_nodes)
    {
        auto body = call_emitter(*node, context);
        if (body)
        {
            auto body_start = writer.position();
            body->serialize(writer);
            writer.align_position(8);
            auto body_size = writer.position() - body_start;
            node_headers.emplace_back(node_header { body->opcode(), (uint32_t)body_size });
        }
    }

    // Write node headers
    auto end_pos = writer.position();
    writer.position(node_headers_pos);
    writer.write_array<node_header>(node_headers);
    writer.position(end_pos);
}

void nncase::codegen::gen_cpp(codegen_context &context, xtl::span<ir::node *> compute_sequence)
{
    std::vector<ir::node *> runtime_nodes;
    std::vector<memory_range> inputs;
    std::vector<runtime_shape_t> input_shapes;
    std::vector<memory_range> outputs;
    std::vector<ir::node *> constants;
    std::vector<runtime_opcode> runtime_opcodes;

    auto name = context.output_path().filename().replace_extension().string();
    auto cpp_out_name = context.output_path();
    std::ofstream cpp_outfile(cpp_out_name, std::ios::binary | std::ios::out);
    if (!cpp_outfile.good())
        throw std::runtime_error("Cannot open file for output: " + cpp_out_name.string());

    auto h_out_name = std::filesystem::path(context.output_path()).replace_extension(".h");
    std::ofstream h_outfile(h_out_name, std::ios::binary | std::ios::out);
    if (!h_outfile.good())
        throw std::runtime_error("Cannot open file for output: " + h_out_name.string());

    binary_writer cpp_writer(cpp_outfile);
    binary_writer h_writer(h_outfile);

    cpp_writer.write_string(fmt::format(R"(// Generated by the nncase compiler.  DO NOT EDIT!

#include <algorithm>
#include <cstdint>
#include "{0}.h"

using namespace nncase;
using namespace nncase::runtime;

namespace {0}
{{
enum kernel_call_result
{{
    kcr_done,
    kcr_async,
    kcr_error
}};

namespace details
{{
    template <class T>
    constexpr T apply_activation(T value, T min, T max)
    {{
        return std::clamp(value, min, max);
    }}
}}

)",
        name));

#define WRITE_CONST(T, dt)                                                                         \
    if (con.output().type() == dt)                                                                 \
    {                                                                                              \
        xtl::span<const T> src(reinterpret_cast<const T *>(data.data()), data.size() / sizeof(T)); \
        for (auto v : src)                                                                         \
        {                                                                                          \
            cpp_writer.write_string(std::to_string(v));                                            \
            cpp_writer.write_string(", ");                                                         \
        }                                                                                          \
    }

    for (auto &&node : compute_sequence)
    {
        if (g_disabled_emitters.find(node->runtime_opcode()) == g_disabled_emitters.end())
            runtime_nodes.emplace_back(node);

        switch (node->runtime_opcode())
        {
        case op_input_node:
            inputs.emplace_back(context.get_allocation(node->output_at(0)));
            input_shapes.emplace_back(ir::to(node->output_at(0).shape()));
            break;
        case op_output_node:
            outputs.emplace_back(context.get_allocation(*node->input_at(0).connection()));
            break;
        case op_constant:
        {
            auto &con = static_cast<constant &>(*node);
            cpp_writer.write_string("const " + std::string(get_type_name(con.output().type())) + "c_" + con.name() + "= { ");
            auto data = con.data();

            WRITE_CONST(float, dt_float32);
            WRITE_CONST(uint8_t, dt_uint8);

            cpp_writer.write_string(" };\n");
            break;
        }
        }
    }

    // write body
    for (auto &&node : runtime_nodes)
    {
        auto body = call_emitter(*node, context);
        if (body)
        {
            runtime_opcodes.emplace_back(body->opcode());
            body->serialize(cpp_writer);
        }
    }

    // opcodes
    cpp_writer.write_string(R"(
runtime_opcode node_runtime_opcode(size_t index)
{
    switch (index)
    {)");

    for (size_t i = 0; i < runtime_opcodes.size(); i++)
    {
        cpp_writer.write_string(fmt::format(R"(
    case {0}:
        return (runtime_opcode){1};)",
            i, runtime_opcodes[i]));
    }

    cpp_writer.write_string(R"(
    default:
        NNCASE_THROW(std::invalid_argument, "Node index is out of range");
    }
}
)");

    // opcodes
    cpp_writer.write_string(R"(
kernel_call_result call_kernel(size_t index, interpreter &interp)
{
    switch (index)
    {)");

    for (size_t i = 0; i < runtime_nodes.size(); i++)
    {
        cpp_writer.write_string(fmt::format(R"(
    case {0}:
        return {1}(interp);)",
            i, runtime_nodes[i]->name()));
    }

    cpp_writer.write_string(R"(
    default:
        NNCASE_THROW(std::invalid_argument, "Node index is out of range");
    }
}
)");

    // step
    cpp_writer.write_string(R"R(
void interpreter::step_()
{
    auto result = kcr_done;

    while (result == kcr_done)
    {
        if (!last_time_)
        {
            last_time_ = clock_t::now();
        }
        else
        {
            auto now = clock_t::now();
            auto duration = now - *last_time_;
            total_duration_ += duration;
            last_time_ = now;

            if (node_profile_)
                node_profile_(last_op_, duration, userdata_);
        }

        if (cnt_node_ == nodes_size())
        {
            run_callback_(userdata_);
            break;
        }
        else
        {
            auto node_id = cnt_node_++;
            auto opcode = node_runtime_opcode(node_id);
            xtl::span<const uint8_t> body(cnt_node_body_, header.body_size);
            cnt_node_body_ += header.body_size;
            last_op_ = opcode;

            result = call_kernel(node_id, *this);

            if (result == kcr_error)
            {
                if (on_error_)
                {
                    char buffer[256];
                    auto name = node_opcode_names(opcode);
                    if (!name.empty())
                        std::sprintf(buffer, "error occurs in running kernel: %s", name.data());
                    else
                        std::sprintf(buffer, "Unknown opcode: (%d)", opcode);
                    on_error_(buffer, userdata_);
                }

                break;
            }
        }
    }
}
)R");

    // input_at
    cpp_writer.write_string(R"(
xtl::span<uint8_t> interpreter::input_at(size_t index)
{
    switch (index)
    {)");

    for (size_t i = 0; i < inputs.size(); i++)
    {
        cpp_writer.write_string(fmt::format(R"(
    case {0}:
        return {{ main_memory_.data() + {1}, {2} }};)",
            i, inputs[i].start, inputs[i].size));
    }

    cpp_writer.write_string(R"(
    default:
        NNCASE_THROW(std::invalid_argument, "Input index is out of range");
    }
}
)");

    // output_at
    cpp_writer.write_string(R"(
xtl::span<uint8_t> interpreter::output_at(size_t index)
{
    switch (index)
    {)");

    for (size_t i = 0; i < outputs.size(); i++)
    {
        cpp_writer.write_string(fmt::format(R"(
    case {0}:
        return {{ main_memory_.data() + {1}, {2} }};)",
            i, outputs[i].start, outputs[i].size));
    }

    cpp_writer.write_string(R"(
    default:
        NNCASE_THROW(std::invalid_argument, "Output index is out of range");
    }
}
)");

    cpp_writer.write_string("}\n");

    h_writer.write_string(fmt::format(R"(// Generated by the nncase compiler.  DO NOT EDIT!

#ifndef NNCASE_INCLUDED_{0}
#define NNCASE_INCLUDED_{0}

#ifdef __cplusplus

#include <algorithm>
#include <array>
#include <cstdint>
#include <cassert>
#include <chrono>
#include <memory>
#include <optional>
#include <runtime/runtime_op.h>
#include <target_config.h>
#include <xtl/xspan.hpp>

namespace {0}
{{
typedef void (*run_callback_t)(void *userdata);
typedef void (*node_profile_callback_t)(nncase::runtime::runtime_opcode op, std::chrono::nanoseconds duration, void *userdata);
typedef void (*error_callback_t)(const char *err, void *userdata);

class interpreter
{{
    using clock_t = std::chrono::system_clock;
public:
    void run(run_callback_t run_callback, node_profile_callback_t node_profile, error_callback_t error_callback, void *userdata);

    uint8_t *main_memory() noexcept {{ return main_memory_.data(); }}
    size_t inputs_size() const noexcept {{ return {2}; }}
    size_t outputs_size() const noexcept {{ return {3}; }}
    size_t nodes_size() const noexcept {{ return {4}; }}

    template <class T>
    xtl::span<T> input_at(size_t index)
    {{
        auto span = input_at(index);
        return {{ reinterpret_cast<T *>(span.data()), span.size() / sizeof(T) }};
    }}

    template <class T>
    xtl::span<T> output_at(size_t index)
    {{
        auto span = output_at(index);
        return {{ reinterpret_cast<T *>(span.data()), span.size() / sizeof(T) }};
    }}

    void step_();

private:
    xtl::span<uint8_t> input_at(size_t index);
    xtl::span<uint8_t> output_at(size_t index);

private:
    alignas(256) std::array<uint8_t, {1}> main_memory_;
    run_callback_t run_callback_;
    node_profile_callback_t node_profile_;
    error_callback_t error_callback_;
    void *userdata_;
    size_t cnt_node_;
    std::chrono::nanoseconds total_duration_;
    std::optional<clock_t::time_point> last_time_;
    nncase::runtime::runtime_opcode last_op_;
}};
}}

#endif
#endif
)",
        name, context.memory_usage(), inputs.size(), outputs.size(), runtime_nodes.size()));
}
