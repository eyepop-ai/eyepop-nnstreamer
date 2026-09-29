/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer tensor_filter, sub-plugin for onnxruntime
 * Copyright (C) 2023 Suyeon Kim <suyeon5.kim@samsung.com>
 */
/**
 * @file        tensor_filter_onnxruntime.cc
 * @date        30 Oct 2023
 * @brief       NNStreamer tensor-filter sub-plugin for ONNXRuntime
 * @see         http://github.com/nnstreamer/nnstreamer
 * @see         https://onnxruntime.ai/
 * @author      Suyeon Kim <suyeon5.kim@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (onnxruntime) for tensor_filter.
 *
 * Every tensor_filter instance is a consumer of a session cache
 * (tensor_filter_onnxruntime_session_cache.hh). Instances in a pipeline that
 * carries a NNS_ONNXRUNTIME_SESSION_CACHE_CONTEXT_TYPE context share one cache;
 * any other instance gets a private cache holding just its own session.
 */

#include <set>
#include <string>
#include <chrono>
#include <optional>
#include <numeric>

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <glib.h>
#include <gmodule.h>
#include <gst/gst.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_onnxruntime_session_cache.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

#include "tensor_filter_onnxruntime_session_cache.hh"

#define ORT_API_MANUAL_INIT 1

#ifdef G_LOG_DOMAIN
#undef G_LOG_DOMAIN
#endif
#define G_LOG_DOMAIN "eyepop-ai"

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

namespace nnstreamer
{
namespace tensor_filter_onnxruntime
{
extern "C" {
void init_filter_onnxruntime (void) __attribute__ ((constructor));
void fini_filter_onnxruntime (void) __attribute__ ((destructor));
G_MODULE_EXPORT gboolean nnstreamer_onnxruntime_session_cache_get_stats (
    NnsSharedSlot *slot, NnsOnnxruntimeSessionCacheStats *stats);
}

// #define DEBUG_TIMING 1

#ifdef DEBUG_TIMING
template<typename TimeDuration = std::chrono::nanoseconds, typename F>
auto timeIt(const F &func, TimeDuration &duration, std::optional<int*> count = std::nullopt) {
  auto start = std::chrono::high_resolution_clock::now();
  auto result = func();
  auto end = std::chrono::high_resolution_clock::now();
  duration += end - start;
  if (count.has_value()) {
    *(count.value()) += 1;
  }
  return result;
}
#define TIME_IT(f, t) timeIt(f, t)
#else
#define TIME_IT(f, t) f()
#endif

typedef enum
{
  cudaMemcpyHostToHost_          =   0,      /**< Host   -> Host */
  cudaMemcpyHostToDevice_        =   1,      /**< Host   -> Device */
  cudaMemcpyDeviceToHost_        =   2,      /**< Device -> Host */
  cudaMemcpyDeviceToDevice_      =   3,      /**< Device -> Device */
  cudaMemcpyDefault_             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
} cudaMemcpyKind_;

typedef enum
{
  cudaStreamCaptureModeGlobal_ = 0,
  cudaStreamCaptureModeThreadLocal_ = 1,
  cudaStreamCaptureModeRelaxed_ = 2
} cudaStreamCaptureMode_;


#define cudaStreamDefault_ 0x00
#define cudaStreamNonBlocking_ 0x01

typedef struct CUstream_st_ * cudaStream_t_;

typedef int (*cudaMalloc_t)(void **p, size_t s);
typedef int (*cudaFree_t)(void *devPtr);
typedef int (*cudaMemcpyAsync_t) (
  void* dst,
  const void* src,
  size_t count,
  cudaMemcpyKind_ kind,
  cudaStream_t_ stream);
typedef int (*cudaStreamCreateWithFlags_t) ( cudaStream_t_* pStream, unsigned int  flags );
typedef int (*cudaStreamDestroy_t) ( cudaStream_t_ stream );
typedef int (*cudaStreamSynchronize_t) ( cudaStream_t_ stream );
typedef int (*cudaThreadExchangeStreamCaptureMode_t) ( cudaStreamCaptureMode_ * mode );
typedef int (*cudaMemGetInfo_t) ( size_t *free, size_t *total );
static cudaMalloc_t cudaMalloc_ = nullptr;
static cudaFree_t cudaFree_ = nullptr;
static cudaMemcpyAsync_t cudaMemcpyAsync_ = nullptr;
static cudaStreamCreateWithFlags_t cudaStreamCreateWithFlags_ = nullptr;
static cudaStreamDestroy_t cudaStreamDestroy_ = nullptr;
static cudaStreamSynchronize_t cudaStreamSynchronize_ = nullptr;
static cudaThreadExchangeStreamCaptureMode_t cudaThreadExchangeStreamCaptureMode_ = nullptr;
static cudaMemGetInfo_t cudaMemGetInfo_ = nullptr;

static gboolean cudaMemcpy_initialized = FALSE;

static void init_cudaMemcpy() {
  if (!cudaMemcpy_initialized) {
    GModule *cuda_module = g_module_open("libcudart.so.13", static_cast<GModuleFlags> (0));
    if (cuda_module == nullptr) {
      g_info("libcudart.so.13 not found, trying libcudart.so.12");
      cuda_module = g_module_open("libcudart.so.12", static_cast<GModuleFlags> (0));
    }
    if (cuda_module == nullptr) {
      g_warning("libcudart.so.13 and libcudart.so.12 not found");
    } else {
      if (!g_module_symbol(
              cuda_module,
              "cudaMalloc",
              reinterpret_cast<gpointer *> (&cudaMalloc_))) {
        g_warning("g_module_symbol( ... \"cudaMalloc\" ...) NOT FOUND");
      }
      if (!g_module_symbol(
              cuda_module,
              "cudaFree",
              reinterpret_cast<gpointer *> (&cudaFree_))) {
        g_warning("g_module_symbol( ... \"cudaFree\" ...) NOT FOUND");
      }
      if (!g_module_symbol(
              cuda_module,
              "cudaMemcpyAsync",
              reinterpret_cast<gpointer *> (&cudaMemcpyAsync_))) {
        g_warning("g_module_symbol( ... \"cudaMemcpyAsync\" ...) NOT FOUND");
      }
      if (!g_module_symbol(
          cuda_module,
          "cudaStreamCreateWithFlags",
          reinterpret_cast<gpointer *> (&cudaStreamCreateWithFlags_))) {
        g_warning("g_module_symbol( ... \"cudaStreamCreateWithFlags_\" ...) NOT FOUND");
      }
      if (!g_module_symbol(
              cuda_module,
              "cudaStreamDestroy",
              reinterpret_cast<gpointer *> (&cudaStreamDestroy_))) {
        g_warning("g_module_symbol( ... \"cudaStreamDestroy\" ...) NOT FOUND");
      }
      if (!g_module_symbol(
              cuda_module,
              "cudaStreamSynchronize",
              reinterpret_cast<gpointer *> (&cudaStreamSynchronize_))) {
        g_warning("g_module_symbol( ... \"cudaStreamSynchronize\" ...) NOT FOUND");
      }
      if (!g_module_symbol(
              cuda_module,
              "cudaThreadExchangeStreamCaptureMode",
              reinterpret_cast<gpointer *> (&cudaThreadExchangeStreamCaptureMode_))) {
        g_warning("g_module_symbol( ... \"cudaThreadExchangeStreamCaptureMode\" ...) NOT FOUND");
      }
      if (!g_module_symbol(
              cuda_module,
              "cudaMemGetInfo",
              reinterpret_cast<gpointer *> (&cudaMemGetInfo_))) {
        g_warning("g_module_symbol( ... \"cudaMemGetInfo\" ...) NOT FOUND");
      }
    }
  }
  cudaMemcpy_initialized = TRUE;
}

struct CudaMemoryDeleter {
  explicit CudaMemoryDeleter(const Ort::Allocator* alloc) noexcept {
    alloc_ = alloc;
  }
  CudaMemoryDeleter(const CudaMemoryDeleter& other) noexcept {
    alloc_ = other.alloc_;
  }
  CudaMemoryDeleter(CudaMemoryDeleter&& other) noexcept {
    alloc_ = other.alloc_;
  }
  void operator()(void* ptr) const {
    ((Ort::Allocator*)alloc_)->Free(ptr);
  }
  const Ort::Allocator* alloc_;
};

#define ORT_LOG_LEVEL "ORT_LOG_LEVEL"
#define ORT_PROVIDER_OPTION_ENV_PREFIX "ORT_PROVIDER_OPTION_"
#define ORT_CONFIG_ENTRY_ENV_PREFIX "ORT_CONFIG_ENTRY_"

static gboolean str_has_prefix_case_insensitive(const gchar *str, const gchar *prefix) {
  if (str == NULL || prefix == NULL) {
    return FALSE;
  }
  gsize prefix_len = strlen(prefix);
  gsize str_len = strlen(str);
  // The string must be at least as long as the prefix
  if (str_len < prefix_len) {
    return FALSE;
  }
  // Compare ignoring case for the length of the prefix
  return g_ascii_strncasecmp(str, prefix, prefix_len) == 0;
}

/** @brief ORT settings read once from the environment of the process. */
struct OrtProcessOptions {
  OrtLoggingLevel log_level = ORT_LOGGING_LEVEL_WARNING;
  std::vector<std::pair<std::string, std::string>> provider_options;
  std::vector<std::pair<std::string, std::string>> config_entries;
};

static const OrtProcessOptions &
process_options ()
{
  static const OrtProcessOptions options = [] () {
    OrtProcessOptions parsed;
    gchar **envp = g_get_environ();
    for (gchar **env_var = envp; *env_var; env_var++) {
      gchar **name_value = g_strsplit(*env_var, "=", 2);
      if (g_ascii_strcasecmp(name_value[0], ORT_LOG_LEVEL) == 0) {
        const gchar* log_level_string = name_value[1];
        g_info("init_ortOptions log level %s=%s", name_value[0], log_level_string);
        if (log_level_string) {
          if (g_ascii_strcasecmp(log_level_string, "FATAL") == 0) {
            parsed.log_level = ORT_LOGGING_LEVEL_FATAL;
          } else if (g_ascii_strcasecmp(log_level_string, "ERROR") == 0) {
            parsed.log_level = ORT_LOGGING_LEVEL_ERROR;
          } else if (g_ascii_strcasecmp(log_level_string, "WARNING") == 0) {
            parsed.log_level = ORT_LOGGING_LEVEL_WARNING;
          } else if (g_ascii_strcasecmp(log_level_string, "INFO") == 0) {
            parsed.log_level = ORT_LOGGING_LEVEL_INFO;
          } else if (g_ascii_strcasecmp(log_level_string, "VERBOSE") == 0) {
            parsed.log_level = ORT_LOGGING_LEVEL_VERBOSE;
          }
        }
      } else if (str_has_prefix_case_insensitive(name_value[0], ORT_PROVIDER_OPTION_ENV_PREFIX)) {
        const gchar *key = name_value[0] + strlen(ORT_PROVIDER_OPTION_ENV_PREFIX);
        const gchar* value = name_value[1];
        g_info("init_ortOptions options %s %s=%s", name_value[0], key, value);
        parsed.provider_options.emplace_back(key, value);
      } else if (str_has_prefix_case_insensitive(name_value[0], ORT_CONFIG_ENTRY_ENV_PREFIX)) {
        const gchar *key = name_value[0] + strlen(ORT_CONFIG_ENTRY_ENV_PREFIX);
        const gchar* value = name_value[1];
        g_info("init_ortOptions config %s %s=%s", name_value[0], key, value);
        parsed.config_entries.emplace_back(key, value);
      }
      g_strfreev(name_value);
    }
    g_strfreev(envp);
    return parsed;
  }();
  return options;
}

/** @brief One Env for every session of the process; never destroyed, sessions may outlive statics. */
static Ort::Env &
shared_env ()
{
  static Ort::Env *env = new Ort::Env (process_options ().log_level, "nnstreamer_onnxruntime");
  return *env;
}

static const gchar *onnx_accl_support[] = { ACCL_CPU_STR, ACCL_GPU_STR, ACCL_NPU_STR, nullptr };

/** @brief The execution provider a session runs on. */
enum class OrtEp { Tensorrt, Cuda, Qnn, Rocm, Openvino, Cpu };

static const char *
ep_name (OrtEp ep)
{
  switch (ep) {
    case OrtEp::Tensorrt: return "tensorrt";
    case OrtEp::Cuda: return "cuda";
    case OrtEp::Qnn: return "qnn";
    case OrtEp::Rocm: return "rocm";
    case OrtEp::Openvino: return "openvino";
    case OrtEp::Cpu: return "cpu";
  }
  return "unknown";
}

/** @brief Everything a session depends on, resolved from a tensor_filter's properties. */
struct SessionPlan {
  std::string model_path;
  OrtEp ep = OrtEp::Cpu;
  int device = 0;
  bool graph = false; /**< CUDA graph capture (TensorRT and CUDA) */
  bool invoke_dynamic = false;
  bool shared = false; /**< the session is shared through a pipeline's cache */
  std::set<std::string> output_names; /**< outputs to bind; empty = all */
  std::string io_contract; /**< input/output formats, static input info, selected outputs */

  bool use_cuda () const
  {
    return ep == OrtEp::Tensorrt || ep == OrtEp::Cuda;
  }
  /** @brief Only one Run() at a time: TensorRT and QNN lock internally, CUDA graphs forbid it. */
  Concurrency concurrency () const
  {
    return ep == OrtEp::Tensorrt || ep == OrtEp::Qnn || (ep == OrtEp::Cuda && graph)
               ? Concurrency::Exclusive : Concurrency::Unbounded;
  }
  /** @brief A graph capture failure falls back to the same session without graph. */
  bool has_fallback () const
  {
    return use_cuda () && graph;
  }
};

/**
 * @brief Internal data structure for tensor information from ONNX.
 */
struct NodeInfo {
  std::size_t count = 0;
  std::vector<std::string> names;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<ONNXTensorElementDataType> types;

  void clear ()
  {
    count = 0;
    names.clear ();
    shapes.clear ();
    types.clear ();
  }
};

class OrtReplica;

/** @brief The binding and device buffers of one Run(); CUDA graphs need them at fixed addresses. */
class IoState
{
  public:
  IoState (OrtReplica &replica, bool own_stream);
  ~IoState ();
  IoState (const IoState &) = delete;
  IoState &operator= (const IoState &) = delete;

  void invoke (GstTensorFilterProperties *prop, const GstTensorMemory *input, GstTensorMemory *output);

  private:
  void prepareInput (const GstTensorMemory *input, GstTensorFilterProperties *prop, std::chrono::nanoseconds &alloc_time, std::chrono::nanoseconds &copy_time);
  void prepareOutput (GstTensorMemory *output, GstTensorFilterProperties *prop, std::chrono::nanoseconds &alloc_time, std::chrono::nanoseconds &copy_time);
  void postProcessInput (GstTensorFilterProperties *prop, std::chrono::nanoseconds &alloc_time, std::chrono::nanoseconds &copy_time);
  void postProcessOutput (GstTensorMemory *output, GstTensorFilterProperties *prop, std::chrono::nanoseconds &alloc_time, std::chrono::nanoseconds &copy_time);
  void clearInputs ();
  void clearOutputs ();

  OrtReplica &replica;
  Ort::IoBinding ioBinding{ nullptr };
  Ort::Allocator allocator{ nullptr };
  cudaStream_t_ cudaStream = nullptr;
  bool owns_stream = false;
  std::vector<Ort::Value> inputTensors;
  std::vector<std::unique_ptr<void, CudaMemoryDeleter>> inputDatas;
  std::vector<Ort::Value> outputTensors;
  std::vector<std::unique_ptr<void, CudaMemoryDeleter>> outputDatas;
};

/** @brief One onnxruntime session, owned by the session cache. */
class OrtReplica : public Replica
{
  public:
  explicit OrtReplica (const SessionPlan &plan);
  ~OrtReplica () override;

  const std::string model_path;
  const bool use_cuda;
  cudaStream_t_ cudaStream = nullptr;
  Ort::Session session{ nullptr };
  Ort::MemoryInfo memInfo{ nullptr };
  Ort::RunOptions runOptions{ nullptr };
  NodeInfo inputNode;
  NodeInfo outputNode;
  std::unique_ptr<IoState> io; /**< set for exclusive replicas, which every consumer runs through */

  private:
  Ort::SessionOptions createSessionOptions (const SessionPlan &plan);
  OrtTensorRTProviderOptionsV2 *createTensorRtOptions (
      const OrtApi &api, const SessionPlan &plan, const gchar *modelDir, gboolean isModelDirWriteable);
};

OrtTensorRTProviderOptionsV2*
OrtReplica::createTensorRtOptions (
  const OrtApi &api, const SessionPlan &plan, const gchar *modelDir,
  gboolean isModelDirWriteable)
{
  OrtTensorRTProviderOptionsV2* options = nullptr;
  Ort::ThrowOnError (api.CreateTensorRTProviderOptions (&options));
  const gchar *trt_fp16_enable = "1";
  const gchar *trt_int8_enable = "0";
  const gchar *trt_engine_cache_enable = "1";
  const gchar *trt_dump_ep_context_model = "0";
  const gchar *trt_ep_context_file_path = nullptr;
  const gchar *trt_engine_cache_path = isModelDirWriteable ? modelDir : g_get_tmp_dir ();
  const auto &provider_options = process_options ().provider_options;
  for (auto iter = provider_options.begin(); iter != provider_options.end(); iter++) {
    if (iter->first.compare("trt_fp16_enable") == 0) {
      trt_fp16_enable = iter->second.c_str ();
    } else if (iter->first.compare("trt_int8_enable") == 0) {
      trt_int8_enable = iter->second.c_str ();
    } else if (iter->first.compare("trt_dump_ep_context_model") == 0) {
      trt_dump_ep_context_model = iter->second.c_str ();
      g_warning("TENSORRT trt_dump_ep_context_model=%s (%lu)", trt_dump_ep_context_model, g_ascii_strtoull(trt_dump_ep_context_model, nullptr, 10));
    } else if (iter->first.compare("trt_ep_context_file_path") == 0) {
      trt_ep_context_file_path = iter->second.c_str ();
    } else if (iter->first.compare("trt_engine_cache_path") == 0) {
      trt_engine_cache_path = iter->second.c_str ();
    }
  }

  std::string device_id = std::to_string (plan.device);
  std::vector<const char *> keys{
    "device_id",
    "trt_onnx_model_folder_path",
    "trt_cuda_graph_enable",
    "trt_fp16_enable",
    "trt_int8_enable",
    "trt_engine_cache_enable",
    "trt_engine_cache_path",
    "trt_timing_cache_enable",
    "trt_timing_cache_path",
  };

  std::vector<const char *> values{
    device_id.c_str (),
    modelDir,
    plan.graph ? "1" : "0",
    trt_fp16_enable,
    trt_int8_enable,
    trt_engine_cache_enable,
    trt_engine_cache_path,
    trt_engine_cache_enable,
    trt_engine_cache_path,
  };

  gchar *file_path = nullptr;
  if (g_ascii_strtoull(trt_dump_ep_context_model, nullptr, 10) > 0) {
    if (trt_ep_context_file_path == nullptr) {
      gchar *model_name = g_path_get_basename(plan.model_path.c_str ());
      if (g_str_has_suffix(model_name, ".onnx")) {
        file_path = g_strdup_printf("%.*s_context.onnx", (int)(strlen(model_name)-strlen(".onnx")), model_name);
      } else {
        file_path = g_strdup_printf("%s.context", model_name);
      }
      g_free(model_name);
      trt_ep_context_file_path = file_path;
    }
    keys.emplace_back("trt_dump_ep_context_model");
    keys.emplace_back("trt_ep_context_file_path");
    values.emplace_back(trt_dump_ep_context_model);
    values.emplace_back(trt_ep_context_file_path);
  }

  Ort::ThrowOnError (api.UpdateTensorRTProviderOptions (
      options, keys.data (), values.data (), keys.size ()));

  g_free(file_path);

  // this implicitly sets "has_user_compute_stream"
  Ort::ThrowOnError (api.UpdateTensorRTProviderOptionsWithValue (
      options, "user_compute_stream", cudaStream));
  return options;
}

/**
 * @brief	The session options of plan; a replica's stream is bound as the
 * compute stream except for CUDA graphs and shared, concurrently run CUDA sessions.
 */
Ort::SessionOptions
OrtReplica::createSessionOptions (const SessionPlan &plan)
{
  Ort::SessionOptions sessionOptions;
  auto api = Ort::GetApi();
  auto device_id = std::to_string (plan.device);

  switch (plan.ep) {
    case OrtEp::Tensorrt: {
      sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

      // ORT's Level-1 WeightBiasQuantization transformer rewrites every fp32 Conv/ConvTranspose/Gemm
      // bias that sits in a QDQ node unit to Cast(int32) -> DequantizeLinear before the graph is
      // partitioned to the EP. TensorRT's IDequantizeLayer accepts only INT8/FP8/INT4/FP4, so an int8
      // QDQ model whose biases are intentionally fp32 (nvidia-modelopt's explicit-precision output)
      // fails engine generation with "Invalid Node - <conv>_bias_dq". QNN HTP wants those int32
      // biases, so only the TensorRT path opts out; an ORT_CONFIG_ENTRY_* env var can still override
      // it on sessionOptions below.
      sessionOptions.AddConfigEntry(kOrtSessionOptionsDisableSpecifiedOptimizers, "WeightBiasQuantization");

      {
        gchar *modelDir = g_path_get_dirname(plan.model_path.c_str ());
        gboolean isModelDirWriteable = access(modelDir, W_OK) == 0;
        // Configure TensorRT Options
        OrtTensorRTProviderOptionsV2* options = createTensorRtOptions(api, plan, modelDir, isModelDirWriteable);
        sessionOptions.AppendExecutionProvider_TensorRT_V2(*options);
        api.ReleaseTensorRTProviderOptions(options);
        g_info("onnxruntime_subplugin::setAccelerator tensorrt graph=%d", plan.graph);
        g_free(modelDir);
      }
      {
        // Standard CUDA provider for ops unsupported by TensorRT
        OrtCUDAProviderOptionsV2* options = nullptr;
        Ort::ThrowOnError(api.CreateCUDAProviderOptions(&options));
        std::vector<const char*> keys{"device_id", "enable_cuda_graph", "cudnn_conv_algo_search"};
        std::vector<const char*> values{device_id.c_str(), plan.graph ? "1" : "0", "HEURISTIC"};
        Ort::ThrowOnError(api.UpdateCUDAProviderOptions(options, keys.data(), values.data(), keys.size()));
        // as a pointer: given as a string, ORT runs every kernel on the legacy default stream
        Ort::ThrowOnError(api.UpdateCUDAProviderOptionsWithValue(options, "user_compute_stream", cudaStream));
        sessionOptions.AppendExecutionProvider_CUDA_V2(*options);
        api.ReleaseCUDAProviderOptions(options);
        g_info("onnxruntime_subplugin::setAccelerator tensorrt: set %ld CUDA options", keys.size());
      }
      break;
    }
    case OrtEp::Cuda: {
      OrtCUDAProviderOptionsV2* options = nullptr;
      Ort::ThrowOnError(api.CreateCUDAProviderOptions(&options));
      std::vector<const char*> keys;
      std::vector<const char*> values;
      if (plan.graph) {
        keys = {"device_id", "enable_cuda_graph", "cudnn_conv_algo_search", "cudnn_conv1d_pad_to_nc1d"};
        values = {device_id.c_str(), "1", "HEURISTIC", "1"};
      } else if (plan.shared) {
        // one compute stream would serialize consumers that Run() this session concurrently
        keys = {"device_id", "enable_cuda_graph", "cudnn_conv_algo_search"};
        values = {device_id.c_str(), "0", "HEURISTIC"};
      } else {
        keys = {"device_id", "enable_cuda_graph", "cudnn_conv_algo_search"};
        values = {device_id.c_str(), "0", "HEURISTIC"};
      }
      Ort::ThrowOnError(api.UpdateCUDAProviderOptions(options, keys.data(), values.data(), keys.size()));
      if (!plan.graph && !plan.shared) {
        // as a pointer: given as a string, ORT runs every kernel on the legacy default stream
        Ort::ThrowOnError(api.UpdateCUDAProviderOptionsWithValue(options, "user_compute_stream", cudaStream));
      }
      sessionOptions.AppendExecutionProvider_CUDA_V2(*options);
      api.ReleaseCUDAProviderOptions(options);
      g_info("onnxruntime_subplugin::setAccelerator cuda: set %ld CUDA options (graph=%d invoke_dynamic=%d)", keys.size(), plan.graph, plan.invoke_dynamic);
      break;
    }
    case OrtEp::Qnn: {
      std::unordered_map<std::string, std::string> options;
      options["backend_type"] = "htp";
      for (const auto &option : process_options ().provider_options) {
        options[option.first] = option.second;
      }
      sessionOptions.AppendExecutionProvider("QNN", options);
      g_info("onnxruntime_subplugin::setAccelerator qnn");
      break;
    }
    case OrtEp::Rocm: {
      OrtROCMProviderOptions* options = nullptr;
      Ort::ThrowOnError(api.CreateROCMProviderOptions(&options));
      sessionOptions.AppendExecutionProvider_ROCM(*options);
      api.ReleaseROCMProviderOptions(options);
      g_info("onnxruntime_subplugin::setAccelerator has_rocm");
      break;
    }
    case OrtEp::Openvino: {
      std::unordered_map<std::string, std::string> options;
      options["device_type"] = "GPU";
      options["enable_qdq_optimizer"] = "True";
      for (const auto &option : process_options ().provider_options) {
        options[option.first] = option.second;
      }
      sessionOptions.AppendExecutionProvider_OpenVINO_V2(options);
      g_info("onnxruntime_subplugin::setAccelerator openvino");
      break;
    }
    case OrtEp::Cpu:
      g_info("onnxruntime_subplugin::setAccelerator NONE");
      break;
  }

  for (const auto& entry : process_options ().config_entries) {
    sessionOptions.AddConfigEntry(entry.first.c_str(), entry.second.c_str());
  }
  // BUG-270: onnxruntime 1.27.x ConstantFolding optimizer seems to produce wrong results
  sessionOptions.AddConfigEntry(kOrtSessionOptionsConstantFoldingMaxOutputSizeInBytes, "0");
  return sessionOptions;
}

OrtReplica::OrtReplica (const SessionPlan &plan)
    : model_path (plan.model_path), use_cuda (plan.use_cuda ())
{
  size_t i, num_inputs, num_outputs;

  if (use_cuda) {
    if (!cudaStreamCreateWithFlags_) {
      throw std::runtime_error ("ERROR creating CUDA stream: libcudart not loaded");
    }
    int cudaError = cudaStreamCreateWithFlags_(&cudaStream, cudaStreamNonBlocking_);
    if (cudaError != 0) {
      cudaStream = nullptr;
      const std::string err_msg
          = "ERROR creating CUDA stream: " + std::to_string(cudaError);
      throw std::runtime_error (err_msg);
    }
  }

  try {
    /* the Env must exist first: provider options log through its default logger */
    Ort::Env &env = shared_env ();
    Ort::SessionOptions sessionOptions = createSessionOptions (plan);
    session = Ort::Session (env, plan.model_path.c_str (), sessionOptions);

    num_inputs = session.GetInputCount ();
    if (num_inputs <= 0 || num_inputs > NNS_TENSOR_SIZE_LIMIT) {
      throw std::invalid_argument (
          std::string ("Too many input tensors: ") + std::to_string (num_inputs)
          + std::string ("max: ") + NNS_TENSOR_SIZE_LIMIT_STR);
    }

    num_outputs = session.GetOutputCount ();
    if (num_outputs <= 0 || num_outputs > NNS_TENSOR_SIZE_LIMIT) {
      throw std::invalid_argument (
          std::string ("Too many output tensors: ") + std::to_string (num_outputs)
          + std::string ("max: ") + NNS_TENSOR_SIZE_LIMIT_STR);
    }
    if (use_cuda) {
      memInfo = Ort::MemoryInfo ("Cuda", OrtAllocatorType::OrtArenaAllocator, plan.device, OrtMemTypeDefault);
      runOptions = Ort::RunOptions ();
      if (!plan.graph) {
        runOptions.AddConfigEntry ("gpu_graph_id", "-1");
      }
    } else {
      memInfo = Ort::MemoryInfo::CreateCpu (
          OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
      runOptions = Ort::RunOptions{ nullptr };
    }

    Ort::AllocatorWithDefaultOptions namesAllocator;

    for (i = 0; i < num_inputs; i++) {
      /* Get input name */
      inputNode.names.emplace_back (session.GetInputNameAllocated (i, namesAllocator).get ());

      /* Get input type and shape */
      Ort::TypeInfo type_info = session.GetInputTypeInfo (i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo ();
      inputNode.types.push_back (tensor_info.GetElementType ());
      inputNode.shapes.push_back (tensor_info.GetShape ());
      inputNode.count++;
    }

    for (i = 0; i < num_outputs; i++) {
      /* Get output name */
      auto output_name = session.GetOutputNameAllocated (i, namesAllocator);
      if (!plan.output_names.empty () && plan.output_names.count (output_name.get ()) == 0) {
        g_info ("skipping model output tensor %s, not pre-configured", output_name.get ());
        continue;
      }
      if (outputNode.count == NNS_TENSOR_SIZE_LIMIT) {
        g_info ("skipping model output tensor %s, max reached", output_name.get ());
        continue;
      }
      outputNode.names.emplace_back (output_name.get ());

      /* Get output type and shape */
      Ort::TypeInfo type_info = session.GetOutputTypeInfo (i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo ();
      outputNode.types.push_back (tensor_info.GetElementType ());
      outputNode.shapes.push_back (tensor_info.GetShape ());

      outputNode.count++;
    }

    if (plan.concurrency () == Concurrency::Exclusive) {
      io = std::make_unique<IoState> (*this, false);
    }
  } catch (...) {
    io.reset ();
    session = Ort::Session{ nullptr };
    if (cudaStream && cudaStreamDestroy_) {
      cudaStreamDestroy_(cudaStream);
    }
    throw;
  }
}

OrtReplica::~OrtReplica ()
{
  io.reset ();
  runOptions = Ort::RunOptions{ nullptr };
  memInfo = Ort::MemoryInfo{ nullptr };
  session = Ort::Session{ nullptr };
  if (cudaStream && cudaStreamDestroy_) {
    cudaStreamDestroy_(cudaStream);
  }
}

IoState::IoState (OrtReplica &replica_, bool own_stream)
    : replica (replica_)
{
  ioBinding = Ort::IoBinding (replica.session);
  if (replica.use_cuda) {
    allocator = Ort::Allocator (replica.session, replica.memInfo);
    if (own_stream) {
      int cudaError = cudaStreamCreateWithFlags_(&cudaStream, cudaStreamNonBlocking_);
      if (cudaError != 0) {
        throw std::runtime_error ("ERROR creating CUDA stream: " + std::to_string(cudaError));
      }
      owns_stream = true;
    } else {
      cudaStream = replica.cudaStream;
    }
  }
}

IoState::~IoState ()
{
  clearInputs ();
  clearOutputs ();
  ioBinding = Ort::IoBinding{ nullptr };
  allocator = Ort::Allocator{ nullptr };
  if (owns_stream && cudaStream && cudaStreamDestroy_) {
    cudaStreamDestroy_(cudaStream);
  }
}

void
IoState::clearInputs ()
{
  if (ioBinding)
    ioBinding.ClearBoundInputs ();
  inputTensors.clear ();
  inputDatas.clear ();
}

void
IoState::clearOutputs ()
{
  if (ioBinding)
    ioBinding.ClearBoundOutputs ();
  outputTensors.clear ();
  outputDatas.clear ();
}

/**
 * @brief Convert the type of tensor.
 * @return 0 if OK. non-zero if error.
 */
static int
convertTensorType (ONNXTensorElementDataType _type, tensor_type &type)
{
  switch (_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      type = _NNS_INT8;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      type = _NNS_UINT8;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      type = _NNS_INT16;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      type = _NNS_UINT16;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      type = _NNS_INT32;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      type = _NNS_UINT32;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      type = _NNS_INT64;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      type = _NNS_UINT64;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      type = _NNS_FLOAT32;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      type = _NNS_FLOAT64;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
#ifdef FLOAT16_SUPPORT
      type = _NNS_FLOAT16;
      break;
#endif
    default:
      nns_loge ("Tensor type not supported: %d", (gint) _type);
      type = _NNS_END;
      return -EINVAL;
  }

  return 0;
}

void
IoState::prepareInput(
    const GstTensorMemory *input, GstTensorFilterProperties *prop, std::chrono::nanoseconds &alloc_time, std::chrono::nanoseconds &copy_time)
{
#ifndef DEBUG_TIMING
  (void)alloc_time;
  (void)copy_time;
#endif
  const NodeInfo &inputNode = replica.inputNode;
  const Ort::MemoryInfo &memInfo = replica.memInfo;
  if (prop == nullptr || prop->input_meta.format == _NNS_TENSOR_FORMAT_STATIC) {
    /* Set input to tensor */
    size_t i;
    if (replica.use_cuda) {
      bool is_reuse_memory = true;
      ioBinding.ClearBoundInputs ();
      if (inputDatas.size () != inputNode.count) {
        inputDatas.clear ();
        is_reuse_memory = false;
      }
      inputTensors.clear ();
      for (i = 0; i < inputNode.count; ++i) {
        auto shape = inputNode.shapes[i].data ();
        auto shape_size = inputNode.shapes[i].size ();
        if (!is_reuse_memory) {
          void* cuda_memory = TIME_IT([&] {
            return allocator.Alloc (input[i].size);
          }, alloc_time);
          inputDatas.emplace_back (std::unique_ptr<void, CudaMemoryDeleter> (
              cuda_memory, CudaMemoryDeleter (&allocator)));
        }
        TIME_IT([&] {
          cudaMemcpyAsync_(inputDatas[i].get (), input[i].data, input[i].size, cudaMemcpyHostToDevice_, cudaStream);
          return nullptr;
        }, copy_time);
        // Create an OrtValue tensor backed by data on CUDA memory
        inputTensors.emplace_back (
            Ort::Value::CreateTensor (memInfo, inputDatas[i].get (),
                input[i].size, shape, shape_size, inputNode.types[i]));
        ioBinding.BindInput(inputNode.names[i].c_str (), inputTensors.back ());
      }
    } else {
      clearInputs ();
      for (i = 0; i < inputNode.count; ++i) {
        auto shape = inputNode.shapes[i].data ();
        auto shape_size = inputNode.shapes[i].size ();
        inputTensors.emplace_back (Ort::Value::CreateTensor (memInfo,
          input[i].data, input[i].size, shape, shape_size, inputNode.types[i]));
        ioBinding.BindInput (inputNode.names[i].c_str (), inputTensors.back ());
      }
    }
  } else if (prop->input_meta.format == _NNS_TENSOR_FORMAT_FLEXIBLE) {
    size_t i;
    clearInputs ();
    if (prop->input_meta.num_tensors > inputNode.count) {
      throw std::runtime_error ("ERROR running model inference: " + std::to_string (prop->input_meta.num_tensors)
          + " input tensors given, model has " + std::to_string (inputNode.count));
    }
    for (i = 0; i < prop->input_meta.num_tensors; i++) {
      std::vector<int64_t> shape;
      GstTensorInfo *tensor_info = gst_tensors_info_get_nth_info (&prop->input_meta, i);
      /* revert order between onnxruntime <> nnstreamer dimensions */
      for (auto j = NNS_TENSOR_RANK_LIMIT - 1; j >= 0; j--) {
        if (tensor_info->dimension[j] > 0) {
          if (tensor_info->dimension[j] == NNS_DIMENSION_ZERO_SIZE) {
            shape.push_back (0);
          } else {
            shape.push_back (tensor_info->dimension[j]);
          }
        }
      }

      if (replica.use_cuda) {
        void* cuda_memory = TIME_IT([&] {
          return allocator.Alloc(input[i].size);
        }, alloc_time);
        inputDatas.emplace_back (std::unique_ptr<void, CudaMemoryDeleter> (
            cuda_memory, CudaMemoryDeleter (&allocator)));
        TIME_IT([&] {
          cudaMemcpyAsync_(inputDatas.back ().get (), input[i].data,
            input[i].size, cudaMemcpyHostToDevice_, cudaStream);
          return nullptr;
        }, copy_time);
        inputTensors.emplace_back (Ort::Value::CreateTensor (memInfo,
            inputDatas.back ().get (), input[i].size, shape.data (),
            shape.size (), inputNode.types[i]));
      } else {
        inputTensors.emplace_back (
            Ort::Value::CreateTensor (memInfo, input[i].data, input[i].size,
                shape.data (), shape.size (), inputNode.types[i]));
      }
      ioBinding.BindInput (inputNode.names[i].c_str (), inputTensors.back ());
    }
  } else {
    const std::string err_msg
        = "ERROR running model inference: does not support tensor format "
          + (std::string) gst_tensor_get_format_string (prop->input_meta.format);
    throw std::runtime_error (err_msg);
  }
}
void
IoState::prepareOutput (GstTensorMemory *output, GstTensorFilterProperties *prop, std::chrono::nanoseconds &alloc_time, std::chrono::nanoseconds &)
{
#ifndef DEBUG_TIMING
  (void)alloc_time;
#endif
  const NodeInfo &outputNode = replica.outputNode;
  const Ort::MemoryInfo &memInfo = replica.memInfo;
  if (prop == nullptr
      || (prop->output_meta.format == _NNS_TENSOR_FORMAT_STATIC && !prop->invoke_dynamic)) {
    /* Set output to tensor */
    size_t i;
    if (replica.use_cuda) {
      if (outputTensors.size () != outputNode.count) {
        clearOutputs ();
        for (i = 0; i < outputNode.count; ++i) {
          void* cuda_memory = TIME_IT([&] {
            return allocator.Alloc (output[i].size);
          }, alloc_time);
          outputDatas.emplace_back (std::unique_ptr<void, CudaMemoryDeleter> (
              cuda_memory, CudaMemoryDeleter (&allocator)));
          // Create an OrtValue tensor backed by data on CUDA memory
          outputTensors.emplace_back (Ort::Value::CreateTensor (memInfo,
              outputDatas.back ().get (), output[i].size,
              outputNode.shapes[i].data (), outputNode.shapes[i].size (),
              outputNode.types[i]));
          ioBinding.BindOutput (outputNode.names[i].c_str (), outputTensors.back ());
        }
      }
    } else {
      clearOutputs ();
      for (i = 0; i < outputNode.count; ++i) {
        outputTensors.emplace_back (Ort::Value::CreateTensor (memInfo,
            output[i].data, output[i].size, outputNode.shapes[i].data (),
            outputNode.shapes[i].size (), outputNode.types[i]));
        ioBinding.BindOutput (outputNode.names[i].c_str (), outputTensors.back ());
      }
    }
  } else {
    size_t i;
    clearOutputs ();
    for (i = 0; i < outputNode.count; ++i) {
      ioBinding.BindOutput (outputNode.names[i].c_str (), memInfo);
    }
  }
}
void
IoState::postProcessInput (GstTensorFilterProperties *prop, std::chrono::nanoseconds &, std::chrono::nanoseconds &)
{
  /* free input tensors asap - keep only when we want to keep static CUDA memory around for the next invocation */
  if (!replica.use_cuda || (prop != nullptr && prop->input_meta.format != _NNS_TENSOR_FORMAT_STATIC)) {
    clearInputs ();
  }
}
void
IoState::postProcessOutput (
    GstTensorMemory *output, GstTensorFilterProperties *prop, std::chrono::nanoseconds &alloc_time, std::chrono::nanoseconds &copy_time)
{
#ifndef DEBUG_TIMING
  (void)alloc_time;
  (void)copy_time;
#endif
  if (prop != nullptr
      && (prop->output_meta.format == _NNS_TENSOR_FORMAT_FLEXIBLE || prop->invoke_dynamic)) {
    size_t i;
    gst_tensors_info_init (&prop->output_meta);
    auto outputValues = ioBinding.GetOutputValues();
    prop->output_meta.num_tensors = outputValues.size ();
    prop->output_meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

    for (i = 0; i < outputValues.size (); i++) {
      size_t scalar_count = 1;
      auto outputInfo = outputValues[i].GetTensorTypeAndShapeInfo ();
      GstTensorInfo *tensor_info = gst_tensors_info_get_nth_info (&prop->output_meta, i);
      if (convertTensorType (outputInfo.GetElementType (), tensor_info->type) != 0) {
        throw std::runtime_error ("Failed to convert ONNX data type.");
      }
      /* revert order between onnxruntime <> nnstreamer dimensions */
      auto rank = outputInfo.GetShape ().size ();
      for (unsigned int shapeI = rank; shapeI > 0; shapeI--) {
        auto dim = outputInfo.GetShape ()[shapeI - 1];
        if (dim == 0) {
          tensor_info->dimension[rank - shapeI] = NNS_DIMENSION_ZERO_SIZE;
        } else {
          tensor_info->dimension[rank - shapeI] = dim;
        }
        scalar_count *= dim;
      }
      output[i].size = scalar_count * gst_tensor_get_element_size (tensor_info->type);
      if (replica.use_cuda) {
        output[i].data = TIME_IT([&] {
          return g_malloc (output[i].size);
        }, alloc_time);
        TIME_IT([&] {
          cudaMemcpyAsync_(output[i].data, outputValues[i].GetTensorRawData (),
            output[i].size, cudaMemcpyDeviceToHost_, cudaStream);
          return nullptr;
        }, copy_time);
      } else
        {
        TIME_IT([&] {
          output[i].data
            = g_memdup2 (outputValues[i].GetTensorRawData (), output[i].size);
          return nullptr;
        }, alloc_time);
      }
    }
    /* free output tensors finally if dynamic output structure */
    clearOutputs ();

  } else if (replica.use_cuda) {
    size_t i;
    for (i = 0; i < outputTensors.size (); i++) {
      TIME_IT([&] {
        cudaMemcpyAsync_(output[i].data, outputDatas[i].get (),
          output[i].size, cudaMemcpyDeviceToHost_, cudaStream);
        return nullptr;
      }, copy_time);
    }
  } else {
    /* free output tensors finally if static output structure and NO cuda */
    clearOutputs ();
  }
}
void
IoState::invoke (GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  std::chrono::nanoseconds alloc_time(std::chrono::nanoseconds::zero());
  std::chrono::nanoseconds copy_time(std::chrono::nanoseconds::zero());
#ifdef DEBUG_TIMING
  std::chrono::nanoseconds total_time(std::chrono::nanoseconds::zero());
  std::chrono::nanoseconds prepare_input_time(std::chrono::nanoseconds::zero());
  std::chrono::nanoseconds prepare_output_time(std::chrono::nanoseconds::zero());
  std::chrono::nanoseconds run_time(std::chrono::nanoseconds::zero());
  std::chrono::nanoseconds post_process_input_time(std::chrono::nanoseconds::zero());
  std::chrono::nanoseconds post_process_output_time(std::chrono::nanoseconds::zero());
  std::chrono::nanoseconds sync_time(std::chrono::nanoseconds::zero());
#endif

  cudaStreamCaptureMode_ mode = cudaStreamCaptureModeThreadLocal_;

  TIME_IT([&]{
    if (replica.use_cuda) {
      cudaThreadExchangeStreamCaptureMode_(&mode);
    }
    TIME_IT([&] {
      prepareInput(input, prop, alloc_time, copy_time);
      if (replica.use_cuda) {
        /* a pageable cudaMemcpy may return before the data lands, and the session
         * runs on a non-blocking stream that does not wait for it: finish the upload */
        cudaStreamSynchronize_(cudaStream);
      }
      return nullptr;
    }, prepare_input_time);
    TIME_IT([&] {
      prepareOutput(output, prop, alloc_time, copy_time); return nullptr;
    }, prepare_output_time);
    TIME_IT([&] {
      replica.session.Run(replica.runOptions, ioBinding); return nullptr;
    }, run_time);
    TIME_IT([&] {
      postProcessInput(prop, alloc_time, copy_time); return nullptr;
    }, post_process_input_time);
    TIME_IT([&] {
      postProcessOutput(output, prop, alloc_time, copy_time); return nullptr;
    }, post_process_output_time);
    if (replica.use_cuda) {
      TIME_IT([&] {
        cudaStreamSynchronize_(cudaStream);
        cudaThreadExchangeStreamCaptureMode_(&mode);
        return nullptr;
      }, sync_time);
    }
    return nullptr;
  }, total_time);

#ifdef DEBUG_TIMING
  std::chrono::nanoseconds unaccounted_time = total_time - (prepare_input_time + prepare_output_time + run_time + post_process_input_time + post_process_output_time) + sync_time;
  g_warning("inference for %s - total: %ld alloc: %ld copy: %ld\n"
            "\t prepare_input: %ld\n"
            "\t prepare_output: %ld\n"
            "\t run_time: %ld\n"
            "\t post_process_input: %ld\n"
            "\t post_process_output: %ld\n"
            "\t sync: %ld\n"
            "\t unaccounted: %ld\n",
    replica.model_path.c_str (), total_time.count(), alloc_time.count(), copy_time.count(),
    prepare_input_time.count(), prepare_output_time.count(),
    run_time.count(),
    post_process_input_time.count(), post_process_output_time.count(),
    sync_time.count(),
    unaccounted_time.count());
#endif
}

/** @brief Bytes in use on a CUDA device. */
static std::optional<uint64_t>
cuda_memory_used (int device)
{
  size_t free_bytes = 0, total_bytes = 0;
  if (device != 0 || !cudaMemGetInfo_ || cudaMemGetInfo_ (&free_bytes, &total_bytes) != 0) {
    return std::nullopt;
  }
  return total_bytes - free_bytes;
}

/** @brief Size of the model file and its external data, the memory estimate for host sessions. */
static uint64_t
model_file_bytes (const std::string &model_path)
{
  uint64_t bytes = 0;
  struct stat st;
  std::vector<std::string> candidates{ model_path, model_path + ".data", model_path + "_data" };
  if (g_str_has_suffix (model_path.c_str (), ".onnx")) {
    candidates.push_back (model_path.substr (0, model_path.size () - strlen (".onnx")) + ".data");
  }
  for (const auto &candidate : candidates) {
    if (stat (candidate.c_str (), &st) == 0 && S_ISREG (st.st_mode)) {
      bytes += st.st_size;
    }
  }
  return bytes;
}

/** @brief Identity of a model file: its real path plus what changes when it is replaced. */
static std::string
model_file_identity (const std::string &model_path)
{
  struct stat st;
  g_autofree gchar *real_path = realpath (model_path.c_str (), nullptr);
  const gchar *path = real_path ? real_path : model_path.c_str ();
  if (stat (path, &st) != 0) {
    return std::string ("path=") + path;
  }
  return std::string ("path=") + path + "|dev=" + std::to_string (st.st_dev)
         + "|ino=" + std::to_string (st.st_ino) + "|size=" + std::to_string (st.st_size)
         + "|mtime=" + std::to_string (st.st_mtim.tv_sec) + "." + std::to_string (st.st_mtim.tv_nsec);
}

static ReplicaSpec
make_spec (const SessionPlan &plan, const std::string &file_identity)
{
  ReplicaSpec spec;
  spec.key = SessionKey (file_identity + "|ep=" + ep_name (plan.ep)
                         + "|device=" + std::to_string (plan.device)
                         + "|graph=" + (plan.graph ? "1" : "0")
                         + "|dynamic=" + (plan.invoke_dynamic ? "1" : "0")
                         + "|" + plan.io_contract);
  spec.concurrency = plan.concurrency ();
  spec.device = plan.use_cuda () ? plan.device : -1;
  if (plan.graph) {
    /* ORT captures on the first Run() of a thread (CUDA EP) or after one warm-up run (TensorRT) */
    spec.serialized_warmup_runs = plan.ep == OrtEp::Tensorrt ? 3 : 2;
  }
  spec.create = [plan] () -> std::unique_ptr<Replica> { return std::make_unique<OrtReplica> (plan); };
  spec.host_bytes = [path = plan.model_path] () { return model_file_bytes (path); };
  return spec;
}

/** @brief Holds a pipeline's cache in its NnsSharedSlot. */
struct SlotCache {
  std::shared_ptr<SessionCache> cache;
};

static void
slot_cache_free (gpointer data)
{
  delete static_cast<SlotCache *> (data);
}

static gpointer
slot_cache_new (gpointer user_data)
{
  /* the slot's destroy notify must never point into an unloaded plugin */
  Dl_info info;
  if (dladdr (reinterpret_cast<void *> (&slot_cache_free), &info) && info.dli_fname) {
    GModule *self = g_module_open (info.dli_fname, static_cast<GModuleFlags> (0));
    if (self) {
      g_module_make_resident (self);
    }
  }
  const CacheConfig &config = *static_cast<const CacheConfig *> (user_data);
  g_info ("onnxruntime session cache: strategy=%s max-sessions=%" G_GUINT64_FORMAT
          " max-bytes=%" G_GUINT64_FORMAT " max-replicas-per-key=%u",
      config.strategy == CacheStrategy::Lru ? "lru" : "asap",
      (guint64) config.max_sessions, (guint64) config.max_bytes, config.max_replicas_per_key);
  return new SlotCache{ SessionCache::create (config, cuda_memory_used) };
}

static bool
structure_get_u64 (const GstStructure *structure, const char *field, uint64_t &value)
{
  const GValue *v = gst_structure_get_value (structure, field);
  if (!v) {
    return false;
  }
  if (G_VALUE_HOLDS_UINT64 (v)) {
    value = g_value_get_uint64 (v);
  } else if (G_VALUE_HOLDS_UINT (v)) {
    value = g_value_get_uint (v);
  } else if (G_VALUE_HOLDS_INT64 (v) && g_value_get_int64 (v) >= 0) {
    value = g_value_get_int64 (v);
  } else if (G_VALUE_HOLDS_INT (v) && g_value_get_int (v) >= 0) {
    value = g_value_get_int (v);
  } else {
    g_warning ("onnxruntime session cache: ignoring field %s of type %s", field, G_VALUE_TYPE_NAME (v));
    return false;
  }
  return true;
}

/** @brief The cache of the pipeline that prop's element belongs to, or nullptr. */
static std::shared_ptr<SessionCache>
pipeline_cache (const GstTensorFilterProperties *prop)
{
  if (!prop->get_context || !prop->context_owner) {
    return nullptr;
  }
  GstContext *context = static_cast<GstContext *> (
      prop->get_context (prop->context_owner, NNS_ONNXRUNTIME_SESSION_CACHE_CONTEXT_TYPE));
  if (!context) {
    return nullptr;
  }

  std::shared_ptr<SessionCache> cache;
  const GstStructure *structure = gst_context_get_structure (context);
  const GValue *slot_value = gst_structure_get_value (structure, NNS_ONNXRUNTIME_SESSION_CACHE_FIELD_SLOT);
  if (!slot_value || !G_VALUE_HOLDS (slot_value, NNS_TYPE_SHARED_SLOT)) {
    g_warning ("onnxruntime session cache: context without a %s field, sessions stay private",
        NNS_ONNXRUNTIME_SESSION_CACHE_FIELD_SLOT);
  } else {
    NnsSharedSlot *slot = static_cast<NnsSharedSlot *> (g_value_get_boxed (slot_value));
    CacheConfig config;
    const gchar *strategy = gst_structure_get_string (structure, NNS_ONNXRUNTIME_SESSION_CACHE_FIELD_STRATEGY);
    if (strategy && g_ascii_strcasecmp (strategy, "lru") == 0) {
      config.strategy = CacheStrategy::Lru;
    } else if (strategy && g_ascii_strcasecmp (strategy, "asap") != 0) {
      g_warning ("onnxruntime session cache: unknown strategy %s, using asap", strategy);
    }
    structure_get_u64 (structure, NNS_ONNXRUNTIME_SESSION_CACHE_FIELD_MAX_SESSIONS, config.max_sessions);
    structure_get_u64 (structure, NNS_ONNXRUNTIME_SESSION_CACHE_FIELD_MAX_BYTES, config.max_bytes);
    uint64_t max_replicas = config.max_replicas_per_key;
    if (structure_get_u64 (structure, NNS_ONNXRUNTIME_SESSION_CACHE_FIELD_MAX_REPLICAS_PER_KEY, max_replicas)) {
      config.max_replicas_per_key = static_cast<unsigned> (std::max<uint64_t> (1, max_replicas));
    }
    auto *holder = static_cast<SlotCache *> (
        nns_shared_slot_get_or_create (slot, slot_cache_new, &config, slot_cache_free));
    if (holder) {
      cache = holder->cache;
    }
  }
  gst_context_unref (context);
  return cache;
}

gboolean
nnstreamer_onnxruntime_session_cache_get_stats (NnsSharedSlot *slot, NnsOnnxruntimeSessionCacheStats *stats)
{
  g_return_val_if_fail (slot != nullptr, FALSE);
  g_return_val_if_fail (stats != nullptr, FALSE);
  auto *holder = static_cast<SlotCache *> (nns_shared_slot_peek (slot));
  if (!holder) {
    return FALSE;
  }
  CacheStats s = holder->cache->stats ();
  stats->entries = s.entries;
  stats->replicas = s.replicas;
  stats->bytes = s.bytes;
  stats->consumers = s.consumers;
  stats->sessions_created = s.sessions_created;
  stats->sessions_destroyed = s.sessions_destroyed;
  stats->hits = s.hits;
  stats->misses = s.misses;
  stats->evictions = s.evictions;
  stats->overcommits = s.overcommits;
  stats->grows = s.grows;
  stats->grow_failures = s.grow_failures;
  stats->poisons = s.poisons;
  return TRUE;
}

/** @brief tensor-filter-subplugin concrete class for onnxruntime */
class onnxruntime_subplugin final : public tensor_filter_subplugin
{
  private:
  bool configured;
  ORTCHAR_T *model_path; /**< The model *.onnx file */

  bool has_tensorrt;
  bool has_cuda;
  bool has_qnn;
  bool has_rocm;
  bool has_openvino;
  accl_hw has_accelerator;

  std::shared_ptr<SessionCache> cache; /**< the pipeline's cache, or a private one */
  bool shared; /**< cache is the pipeline's */
  EntryRef entry;
  SessionKey primary_key;
  std::optional<ReplicaSpec> fallback;

  NodeInfo inputNode;
  NodeInfo outputNode;

  /** @brief This consumer's binding on an unbounded (concurrently run) replica. */
  std::unique_ptr<IoState> consumer_io;
  const OrtReplica *consumer_io_replica;

  static const char *name;
  static onnxruntime_subplugin *registeredRepresentation;

  void cleanup ();
  void convertTensorInfo (const NodeInfo &node, GstTensorsInfo &info);
  int convertTensorDim (const std::vector<int64_t> &shapes, tensor_dim &dim, bool &is_dynamic);
  SessionPlan resolvePlan (const GstTensorFilterProperties *prop, bool shared);
  void invokeOn (Lease &lease, GstTensorFilterProperties *prop,
      const GstTensorMemory *input, GstTensorMemory *output);

  public:
  static void init_filter_onnxruntime ();
  static void fini_filter_onnxruntime ();

  onnxruntime_subplugin ();
  ~onnxruntime_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void invoke_dynamic (GstTensorFilterProperties *prop,
      const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

/**
 * @brief Constructor for onnxruntime_subplugin.
 */
onnxruntime_subplugin::onnxruntime_subplugin ()
    : configured{ false }, model_path{ nullptr },
      has_tensorrt{ false },
      has_cuda{ false }, has_qnn{ false },
      has_rocm{ false }, has_openvino{ false },
      has_accelerator{ ACCL_NONE },
      shared{ false },
      consumer_io_replica{ nullptr }
{
  process_options ();
  std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
  for (auto provider_name : availableProviders) {
    if (provider_name == "CUDAExecutionProvider") {
      init_cudaMemcpy();
      has_cuda = true;
    } else if (provider_name == "TensorrtExecutionProvider") {
        init_cudaMemcpy();
        has_tensorrt = true;
      } else if (provider_name == "ROCMExecutionProvider") {
      has_rocm = true;
    } else if (provider_name == "QNNExecutionProvider") {
      has_qnn = true;
    } else if (provider_name == "OpenVINOExecutionProvider") {
      has_openvino = true;

    }
  }

  nns_logi("onnxruntime provider: tensorrt=%d cuda=%d rocm=%d, qnn=%d, openvino=%d",
      has_tensorrt, has_cuda, has_rocm, has_qnn, has_openvino);
}

/**
 * @brief Destructor for onnxruntime_subplugin.
 */
onnxruntime_subplugin::~onnxruntime_subplugin ()
{
  cleanup ();
}

/** @brief cleanup resources used by onnxruntime subplugin */
void
onnxruntime_subplugin::cleanup ()
{
  if (!configured)
    return; /* Nothing to do if it is an empty model */

  /* the binding belongs to a session of the entry, release it first */
  consumer_io.reset ();
  consumer_io_replica = nullptr;
  entry.reset ();
  cache.reset ();
  fallback.reset ();

  inputNode.clear ();
  outputNode.clear ();

  g_free (model_path);
  model_path = nullptr;
  configured = false;
}

/**
 * @brief Convert ONNX tensor information.
 */
void
onnxruntime_subplugin::convertTensorInfo (const NodeInfo &node, GstTensorsInfo &info)
{
  GstTensorInfo *_info;
  gst_tensors_info_init (std::addressof (info));
  info.num_tensors = (unsigned int) node.count;
  for (guint i = 0; i < info.num_tensors; ++i) {
    bool is_dynamic = false;
    _info = gst_tensors_info_get_nth_info (std::addressof (info), i);

    if (convertTensorType (node.types[i], _info->type) != 0)
      throw std::runtime_error ("Failed to convert ONNX data type.");

    if (convertTensorDim (node.shapes[i], _info->dimension, is_dynamic) != 0)
      throw std::runtime_error ("Failed to convert ONNX shape.");
    _info->name = g_strdup (node.names[i].c_str ());
    if (is_dynamic) {
      info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
    }
  }
}

/**
 * @brief Convert the shape of tensor.
 * @return 0 if OK. non-zero if error.
 */
int
onnxruntime_subplugin::convertTensorDim (const std::vector<int64_t> &shapes, tensor_dim &dim, bool &is_dynamic)
{
  size_t i, rank;
  is_dynamic = false;
  rank = shapes.size ();
  if (rank > NNS_TENSOR_RANK_LIMIT) {
    nns_loge("Invalid shape (rank %zu, max: %d)", rank, NNS_TENSOR_RANK_LIMIT);
    return -EINVAL;
  }
  if (rank == 0) {
    // scalar value, simulate with shape [1]
    rank = 1;
    dim[0] = 1;
  } else {
    /* the order of dimension is reversed at CAPS negotiation */
    for (i = 0; i < rank; i++) {
      /* free dimensions are treated as 1 if not overridden */
      if (shapes[rank - i - 1] < 0) {
        is_dynamic = true;
        dim[i] = 1;
      } else {
        dim[i] = shapes[rank - i - 1];
      }
    }
  }

  /* fill remaining entries with 0 */
  for (i = rank; i < NNS_TENSOR_RANK_LIMIT; ++i) {
    dim[i] = 0;
  }

  return 0;
}

/**
 * @brief Method to get empty object.
 */
tensor_filter_subplugin &
onnxruntime_subplugin::getEmptyInstance ()
{
  return *(new onnxruntime_subplugin ());
}

/**
 * @brief	Resolves the accelerator, custom properties and I/O of prop into a session plan.
 */
SessionPlan
onnxruntime_subplugin::resolvePlan (const GstTensorFilterProperties *prop, bool shared)
{
  bool enable_tensorrt = false;
  bool disable_cuda = false;
  bool disable_cuda_graph = false;

  nns_logi("num_hw: %d acc string: %s", prop->num_hw, prop->accl_str);
  for (int j = 0; j < prop->num_hw; j++) {
    nns_logi("prop->hw_list[i]: %d", prop->hw_list[j]);
  }

  if (prop->custom_properties) {
    g_info("onnxruntime_subplugin::configure_instance prop->custom_properties=%s", prop->custom_properties);
    gchar** custom_properties = g_strsplit(prop->custom_properties, ";", 0);
    for (int i = 0; custom_properties[i]; i++) {
      if (strcmp(custom_properties[i], "disable_cuda") == 0) {
        disable_cuda = true;
      } else if (strcmp(custom_properties[i], "enable_tensorrt") == 0) {
        enable_tensorrt = true;
      } else if (strcmp(custom_properties[i], "disable_cuda_graph") == 0) {
        disable_cuda_graph = true;
      }
      /* disable_qnn, disable_rocm and disable_openvino are accepted but have never selected anything */
    }
    g_strfreev(custom_properties);
  }

  SessionPlan plan;
  plan.model_path = model_path;
  plan.shared = shared;
  plan.invoke_dynamic = prop->invoke_dynamic ||
                        prop->input_meta.format != _NNS_TENSOR_FORMAT_STATIC ||
                        prop->output_meta.format != _NNS_TENSOR_FORMAT_STATIC;

  accl_hw use_accelerator = parse_accl_hw (prop->accl_str, onnx_accl_support, nullptr, nullptr);
  g_info("onnxruntime_subplugin::setAccelerator(%s) CPU=%d GPU=%d NPU=%d", prop->accl_str, use_accelerator & ACCL_CPU, use_accelerator & ACCL_GPU, use_accelerator & ACCL_NPU);
  if (has_tensorrt && (use_accelerator & ACCL_GPU) && enable_tensorrt) {
    plan.ep = OrtEp::Tensorrt;
    plan.graph = !disable_cuda_graph;
  } else if (has_cuda && (use_accelerator & ACCL_GPU) && !disable_cuda) {
    plan.ep = OrtEp::Cuda;
    plan.graph = !disable_cuda_graph && !plan.invoke_dynamic;
  } else if (has_qnn && (use_accelerator & (ACCL_NPU | ACCL_GPU))) {
    plan.ep = OrtEp::Qnn;
  } else if (has_rocm && (use_accelerator & ACCL_GPU)) {
    plan.ep = OrtEp::Rocm;
  } else if (has_openvino && (use_accelerator & ACCL_GPU)) {
    plan.ep = OrtEp::Openvino;
  } else {
    plan.ep = OrtEp::Cpu;
  }

  for (guint i = 0; i < prop->output_meta.num_tensors; i++) {
    GstTensorInfo *tensor_info = gst_tensors_info_get_nth_info (
        const_cast<GstTensorsInfo *> (&prop->output_meta), i);
    if (tensor_info->name) {
      plan.output_names.insert (tensor_info->name);
    }
  }

  g_autofree gchar *input_info = gst_tensors_info_to_string (&prop->input_meta);
  std::string outputs;
  for (const auto &output_name : plan.output_names) {
    outputs += (outputs.empty () ? "" : ",") + output_name;
  }
  plan.io_contract = std::string ("in=") + gst_tensor_get_format_string (prop->input_meta.format)
                     + ":" + (input_info ? input_info : "")
                     + "|out=" + gst_tensor_get_format_string (prop->output_meta.format)
                     + ":" + outputs;
  return plan;
}

/**
 * @brief Method to prepare/configure onnxruntime instance.
 */
void
onnxruntime_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  if (configured) {
    /* Already opened */
    if (!prop->model_files[0] || prop->model_files[0][0] == '\0') {
      throw std::runtime_error ("Model path is not given.");
    }
    cleanup ();
  }

  if (!g_file_test (prop->model_files[0], G_FILE_TEST_IS_REGULAR)) {
    const std::string err_msg
        = "Given file " + (std::string) prop->model_files[0] + " is not valid";
    cleanup ();
    throw std::runtime_error (err_msg);
  }

  // Handle Windows path conversion
#if (defined(_WIN32) || defined(__CYGWIN__))
  // TODO: add error checking and check type of model_files
  char *model_path_char = g_strdup (prop->model_files[0]);

  int wlen = mbstowcs(NULL, model_path_char, 0);
  model_path = (wchar_t*) malloc((wlen + 1) * sizeof(wchar_t));

  mbstowcs(model_path, model_path_char, wlen + 1);
  g_free(model_path_char);
#else
  model_path = g_strdup (prop->model_files[0]);
#endif

  cache = pipeline_cache (prop);
  shared = cache != nullptr;
  if (!cache) {
    cache = SessionCache::create (CacheConfig{}, cuda_memory_used);
  }

  SessionPlan plan = resolvePlan (prop, shared);
  const std::string file_identity = model_file_identity (plan.model_path);
  ReplicaSpec primary = make_spec (plan, file_identity);
  primary_key = primary.key;
  if (plan.has_fallback ()) {
    SessionPlan fallback_plan = plan;
    fallback_plan.graph = false;
    fallback = make_spec (fallback_plan, file_identity);
  }

  try {
    entry = cache->acquire (primary, fallback ? &*fallback : nullptr);
    entry.inspect ([this] (const Replica &replica) {
      const auto &r = static_cast<const OrtReplica &> (replica);
      inputNode = r.inputNode;
      outputNode = r.outputNode;
    });
  } catch (const Ort::Exception &exception) {
    entry.reset ();
    cache.reset ();
    fallback.reset ();
    g_free (model_path);
    model_path = nullptr;
    throw std::runtime_error ("ERROR running model inference: " + (std::string) exception.what ());
  } catch (...) {
    entry.reset ();
    cache.reset ();
    fallback.reset ();
    g_free (model_path);
    model_path = nullptr;
    throw;
  }
  configured = true;
}

/**
 * @brief Runs one inference on a leased replica, through the replica's binding
 * (exclusive) or this consumer's own (unbounded).
 */
void
onnxruntime_subplugin::invokeOn (Lease &lease, GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  auto *replica = static_cast<OrtReplica *> (lease.replica ());
  IoState *io = replica->io.get ();
  if (!io) {
    if (consumer_io_replica != replica) {
      consumer_io.reset ();
      /* a shared session runs consumers concurrently, each copying on its own stream */
      consumer_io = std::make_unique<IoState> (*replica, shared);
      consumer_io_replica = replica;
    }
    io = consumer_io.get ();
  }
  lease.run ([&] () { io->invoke (prop, input, output); });
}

/**
 * @brief Method to execute the model with dynamic tensors.
 */
void
onnxruntime_subplugin::invoke_dynamic (GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  g_assert (configured);

  if (!input)
    throw std::runtime_error ("Invalid input buffer, it is NULL.");
  if (!output)
    throw std::runtime_error ("Invalid output buffer, it is NULL.");

  try {
    Lease lease = entry.lease ();
    invokeOn (lease, prop, input, output);
  } catch (const Ort::Exception &exception) {
    if (fallback && entry.spec ().key == primary_key) {
      g_info ("ONNX provider error '%s' for %s, trying to use fallback session options",
          exception.what (), model_path);
      try {
        consumer_io.reset ();
        consumer_io_replica = nullptr;
        entry.poison (*fallback);
        Lease lease = entry.lease ();
        invokeOn (lease, prop, input, output);
      } catch (const Ort::Exception &exception) {
        const std::string err_msg
            = "ERROR running model inference: " + (std::string) exception.what ();
        throw std::runtime_error (err_msg);
      }
    } else {
      g_info ("ONNX provider error '%s' for %s, no fallback",
        exception.what (), model_path);

      const std::string err_msg
          = "ERROR running model inference: " + (std::string) exception.what ();
      throw std::runtime_error (err_msg);
    }
  }
}

/**
 * @brief Method to execute the model.
 */
void
onnxruntime_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  invoke_dynamic(nullptr, input, output);
}

/**
 * @brief Method to get the information of onnxruntime subplugin.
 */
void
onnxruntime_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = 0;
  info.allocate_in_invoke = 0;
  info.run_without_model = 0;
  info.verify_model_path = 1;
  if (has_cuda || has_qnn || has_openvino || has_rocm) {
    has_accelerator = ACCL_GPU;
    info.num_hw = 1;
    info.hw_list = &has_accelerator;
    info.accl_auto = has_accelerator;
    info.accl_default = has_accelerator;
  }
}

/**getInputDimension
 * @brief Method to get the model information.
 */
int
onnxruntime_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops == GET_IN_OUT_INFO) {
    convertTensorInfo (inputNode, in_info);
    convertTensorInfo (outputNode, out_info);

    /* For debug, print input and output tensor information. */
    g_autofree gchar *instr = gst_tensors_info_to_string (std::addressof (in_info));
    g_autofree gchar *outstr = gst_tensors_info_to_string (std::addressof (out_info));
    nns_logd ("Input info: %s", instr);
    nns_logd ("Output info: %s", outstr);

    return 0;
  }

  return -ENOENT;
}

/**
 * @brief Method to handle events.
 */
int
onnxruntime_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  if (ops == SET_CONTEXT && !shared && data.context_type
      && g_str_equal (data.context_type, NNS_ONNXRUNTIME_SESSION_CACHE_CONTEXT_TYPE)) {
    /* opened before the pipeline's cache was known; reopen to join it */
    return 0;
  }
  return -ENOENT;
}

const char *onnxruntime_subplugin::name = "onnxruntime";
onnxruntime_subplugin *onnxruntime_subplugin::registeredRepresentation = nullptr;

/** @brief Initialize this object for tensor_filter subplugin runtime register. */
void
onnxruntime_subplugin::init_filter_onnxruntime ()
{
  Ort::InitApi();
  std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
  for (auto f : availableProviders) {
    nns_logi("onnxruntime %s filter found provider: %s",  Ort::GetVersionString().c_str(), f.c_str());
  }

  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<onnxruntime_subplugin> ();
}

/** @brief Destruct the sub-plugin for onnxruntime. */
void
onnxruntime_subplugin::fini_filter_onnxruntime ()
{
  g_assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/** @brief initializer */
void
init_filter_onnxruntime ()
{
  if (nnstreamer_filter_find ("onnx")) {
    nns_loge ("Cannot use onnxruntime and onnx both. Won't register this onnxruntime subplugin.");
    return;
  }

  onnxruntime_subplugin::init_filter_onnxruntime ();
}

/** @brief finalizer */
void
fini_filter_onnxruntime ()
{
  onnxruntime_subplugin::fini_filter_onnxruntime ();
}

} /* namespace tensor_filter_onnxruntime */
} /* namespace nnstreamer */
