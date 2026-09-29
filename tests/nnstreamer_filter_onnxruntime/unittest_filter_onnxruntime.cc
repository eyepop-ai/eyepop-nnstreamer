/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_onnxruntime.cc
 * @date    30 Oct 2023
 * @brief   Unit test for onnxruntime tensor filter sub-plugin
 * @author  Suyeon Kim <suyeon5.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */

#include <gtest/gtest.h>
#include <vector>
#include <glib.h>
#include <gst/gst.h>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_util.h>
#include <tensor_common.h>
#include <unittest_util.h>
#include "nnstreamer_plugin_api.h"
#include "nnstreamer_plugin_api_util.h"
#include <nnstreamer_onnxruntime_session_cache.h>

/**
 * @brief internal function to get model filename
 */
static gboolean
_GetModelFilePath (gchar **model_file)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  g_autofree gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();
  std::string model_name = "mobilenet_v2_quant.onnx";

  *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", model_name.c_str (), NULL);

  return g_file_test (*model_file, G_FILE_TEST_EXISTS);
}

/**
 * @brief internal function to get the orange.png
 */
static gboolean
_GetOrangePngFilePath (gchar **input_file)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  g_autofree gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();
  std::string input_file_name = "orange.png";

  *input_file = g_build_filename (
      root_path, "tests", "test_models", "data", input_file_name.c_str (), NULL);

  return g_file_test (*input_file, G_FILE_TEST_EXISTS);
}

/**
 * @brief Set tensor filter properties
 */
static void
_SetFilterProp (GstTensorFilterProperties *prop, const gchar *name, const gchar **models)
{
  memset (prop, 0, sizeof (GstTensorFilterProperties));
  prop->fwname = name;
  prop->fw_opened = 0;
  prop->model_files = models;
  prop->num_models = g_strv_length ((gchar **) models);
}

/**
 * @brief Signal to validate the result in tensor_sink
 */
static void
check_output (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  GstMemory *mem_res;
  GstMapInfo info_res;
  gboolean mapped;
  UNUSED (element);

  mem_res = gst_buffer_get_memory (buffer, 0);
  mapped = gst_memory_map (mem_res, &info_res, GST_MAP_READ);
  ASSERT_TRUE (mapped);

  gint is_float = (gint) * ((guint8 *) user_data);
  guint idx, max_idx = 0U;

  if (is_float == 0) {
    guint8 *output = (guint8 *) info_res.data;
    guint8 max_value = 0;

    for (idx = 0; idx < info_res.size; ++idx) {
      if (output[idx] > max_value) {
        max_value = output[idx];
        max_idx = idx;
      }
    }
  } else if (is_float == 1) {
    gfloat *output = (gfloat *) info_res.data;
    gfloat max_value = G_MINFLOAT;

    for (idx = 0; idx < (info_res.size / sizeof (gfloat)); ++idx) {
      if (output[idx] > max_value) {
        max_value = output[idx];
        max_idx = idx;
      }
    }
  }

  gst_memory_unmap (mem_res, &info_res);
  gst_memory_unref (mem_res);

  EXPECT_EQ (max_idx, 951U);
}

/**
 * @brief Negative test case with invalid model file path
 */
TEST (nnstreamerFilterOnnxRuntime, openClose00)
{
  int ret;
  void *data = NULL;

  const gchar *model_files[] = {
    "some/invalid/model/path.onnx",
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("onnxruntime");
  EXPECT_NE (sp, nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "onnxruntime", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Positive case with successful getModelInfo
 */
TEST (nnstreamerFilterOnnxRuntime, getModelInfo00)
{
  int ret;
  void *data = NULL;
  gchar *model_file;

  ASSERT_TRUE (_GetModelFilePath (&model_file));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("onnxruntime");
  EXPECT_NE (sp, nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "onnxruntime", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
}

/**
 * @brief Positive case with successful getModelInfo
 */
TEST (nnstreamerFilterOnnxRuntime, getModelInfo00_1)
{
  int ret;
  void *data = NULL;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("onnxruntime");
  EXPECT_NE (sp, nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "onnxruntime", model_files);

  ret = sp->open (&prop, &data);

  EXPECT_EQ (ret, 0);

  GstTensorsInfo in_info, out_info;
  ret = sp->getModelInfo (NULL, NULL, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (in_info.num_tensors, 1U);
  EXPECT_EQ (in_info.info[0].dimension[0], 224U);
  EXPECT_EQ (in_info.info[0].dimension[1], 224U);
  EXPECT_EQ (in_info.info[0].dimension[2], 3U);
  EXPECT_EQ (in_info.info[0].dimension[3], 1U);
  EXPECT_EQ (in_info.info[0].type, _NNS_FLOAT32);

  EXPECT_EQ (out_info.num_tensors, 1U);
  EXPECT_EQ (out_info.info[0].dimension[0], 1000U);
  EXPECT_EQ (out_info.info[0].dimension[1], 1U);
  EXPECT_EQ (out_info.info[0].dimension[2], 0U);
  EXPECT_EQ (out_info.info[0].dimension[3], 0U);
  EXPECT_EQ (out_info.info[0].type, _NNS_FLOAT32);

  sp->close (&prop, &data);

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Test onnxruntime subplugin with successful invoke for sample onnx model (input data type: float)
 */
TEST (nnstreamerFilterOnnxRuntime, invoke00)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("onnxruntime");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "onnxruntime", model_files);

  input.size = sizeof (float) * 224 * 224 * 3 * 1;
  output.size = sizeof (float) * 1000 * 1;

  input.data = g_malloc0 (input.size);
  output.data = g_malloc0 (output.size);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /* invoke successful */
  ret = sp->invoke (NULL, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);

  g_free (input.data);
  g_free (output.data);

  sp->close (&prop, &data);
}

/**
 * @brief Negative case with invalid input/output
 */
TEST (nnstreamerFilterOnnxRuntime, invoke01_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("onnxruntime");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "onnxruntime", model_files);

  output.size = input.size = sizeof (float) * 1;
  input.data = g_malloc0 (input.size);
  output.data = g_malloc0 (output.size);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /* catching exception */
  EXPECT_NE (sp->invoke (NULL, &prop, data, NULL, &output), 0);
  EXPECT_NE (sp->invoke (NULL, &prop, data, &input, NULL), 0);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case to launch gst pipeline: wrong dimension
 */
TEST (nnstreamerFilterOnnxRuntime, launch00_n)
{
  GstElement *gstpipe;
  GError *err = NULL;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file));

  /* create a nnstreamer pipeline */
  g_autofree gchar *pipeline = g_strdup_printf (
      "videotestsrc num-buffers=10 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=42,height=42,framerate=0/1 ! tensor_converter ! tensor_filter framework=onnxruntime model=\"%s\" latency=1 ! tensor_sink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
}

/**
 * @brief Negative case to launch gst pipeline: wrong data type
 */
TEST (nnstreamerFilterOnnxRuntime, launch01_n)
{
  GstElement *gstpipe;
  GError *err = NULL;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file));

  /* create a nnstreamer pipeline */
  g_autofree gchar *pipeline = g_strdup_printf (
      "videotestsrc num-buffers=10 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_filter framework=onnxruntime model=\"%s\" latency=1 ! tensor_sink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
}

/**
 * @brief Positive case to launch gst pipeline
 */
TEST (nnstreamerFilterOnnxRuntime, floatModelResult)
{
  GstElement *gstpipe;
  GError *err = NULL;
  g_autofree gchar *model_file = NULL;
  g_autofree gchar *input_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file));
  ASSERT_TRUE (_GetOrangePngFilePath (&input_file));

  /* create a nnstreamer pipeline */
  g_autofree gchar *pipeline = g_strdup_printf (
      "filesrc location=\"%s\" ! pngdec ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_transform mode=arithmetic option=typecast:float32,div:127.5,add:-1.0 ! tensor_filter framework=onnxruntime model=\"%s\" ! tensor_sink name=sink",
      input_file, model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  GstElement *sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sink");

  ASSERT_TRUE (sink_handle != nullptr);

  guint8 is_float = 1;

  g_signal_connect (sink_handle, "new-data", (GCallback) check_output, &is_float);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT * 10),
      0);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (gstpipe);
}

/**
 * @brief Negative case with incorrect path
 */
TEST (nnstreamerFilterOnnxRuntime, error00_n)
{
  GstElement *gstpipe;
  GError *err = NULL;
  int status = 0;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  g_autofree gchar *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "incorrect_path.onnx", NULL);

  /* Create a nnstreamer pipeline */
  g_autofree gchar *pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter framework=onnxruntime model=\"%s\" ! fakesink",
      model_file);
  gstpipe = gst_parse_launch (pipeline, &err);

  if (gstpipe) {
    EXPECT_NE (gst_element_set_state (gstpipe, GST_STATE_PLAYING), GST_STATE_CHANGE_SUCCESS);
    EXPECT_EQ (gst_element_set_state (gstpipe, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
    g_usleep (500000);
    EXPECT_NE (gst_element_set_state (gstpipe, GST_STATE_NULL), GST_STATE_CHANGE_FAILURE);
    g_usleep (100000);

    gst_object_unref (gstpipe);
  } else {
    status = -1;
    ml_loge ("GST PARSE LAUNCH FAILED: [%s], %s\n", pipeline,
        (err) ? err->message : "unknown reason");
    g_clear_error (&err);
  }
  EXPECT_EQ (status, 0);
}

/**
 * @brief Negative case with incorrect tensor meta
 */
TEST (nnstreamerFilterOnnxRuntime, error01_n)
{
  GstElement *gstpipe;
  GError *err = NULL;
  int status = 0;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file));

  /* Create a nnstreamer pipeline */
  g_autofree gchar *pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=240,height=224 ! tensor_converter ! tensor_filter framework=onnxruntime model=\"%s\" ! fakesink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  if (gstpipe) {
    GstState state, pending;

    EXPECT_NE (gst_element_set_state (gstpipe, GST_STATE_PLAYING), GST_STATE_CHANGE_SUCCESS);
    g_usleep (500000);
    /* This should fail: dimension mismatched. */
    EXPECT_EQ (gst_element_get_state (gstpipe, &state, &pending, GST_SECOND / 4),
        GST_STATE_CHANGE_FAILURE);

    EXPECT_NE (gst_element_set_state (gstpipe, GST_STATE_NULL), GST_STATE_CHANGE_FAILURE);
    g_usleep (100000);

    gst_object_unref (gstpipe);
  } else {
    status = -1;
    ml_loge ("GST PARSE LAUNCH FAILED: [%s], %s\n", pipeline,
        (err) ? err->message : "unknown reason");
    g_clear_error (&err);
  }
  EXPECT_EQ (status, 0);
}

/**
 * @brief A session cache context for a pipeline, around a fresh slot.
 */
static GstContext *
_NewSessionCacheContext (NnsSharedSlot *slot, const gchar *strategy)
{
  GstContext *context = gst_context_new (NNS_ONNXRUNTIME_SESSION_CACHE_CONTEXT_TYPE, TRUE);
  GstStructure *structure = gst_context_writable_structure (context);
  gst_structure_set (structure, NNS_ONNXRUNTIME_SESSION_CACHE_FIELD_SLOT,
      NNS_TYPE_SHARED_SLOT, slot, NNS_ONNXRUNTIME_SESSION_CACHE_FIELD_STRATEGY,
      G_TYPE_STRING, strategy, NULL);
  return context;
}

/**
 * @brief Reads the stats of the cache in slot through the symbol the sub-plugin exports.
 */
static gboolean
_GetSessionCacheStats (NnsSharedSlot *slot, NnsOnnxruntimeSessionCacheStats *stats)
{
  const gchar *filters = g_getenv ("NNSTREAMER_FILTERS");
  g_autofree gchar *path = g_build_filename (
      filters ? filters : ".", "libnnstreamer_filter_onnxruntime.so", NULL);
  GModule *module = g_module_open (path, G_MODULE_BIND_LAZY);
  NnsOnnxruntimeSessionCacheGetStatsFunc get_stats = NULL;
  gboolean found = FALSE;

  if (!module)
    return FALSE;
  if (g_module_symbol (module, NNS_ONNXRUNTIME_SESSION_CACHE_GET_STATS_SYMBOL, (gpointer *) &get_stats))
    found = get_stats (slot, stats);
  g_module_close (module);
  return found;
}

/**
 * @brief Plays pipeline until EOS, leaving it in PLAYING.
 */
static void
_PlayToEos (GstElement *pipeline, guint timeout_s = 10)
{
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, timeout_s * 1000U), 0);
  GstBus *bus = gst_element_get_bus (pipeline);
  GstMessage *msg = gst_bus_timed_pop_filtered (
      bus, timeout_s * GST_SECOND, (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  ASSERT_TRUE (msg != nullptr);
  EXPECT_EQ (GST_MESSAGE_TYPE (msg), GST_MESSAGE_EOS);
  gst_message_unref (msg);
  gst_object_unref (bus);
}

/**
 * @brief A pipeline classifying the orange with the model, through filter_desc per branch.
 */
static GstElement *
_NewOrangePipeline (const gchar *branch0, const gchar *branch1)
{
  g_autofree gchar *model_file = NULL;
  g_autofree gchar *input_file = NULL;
  GError *err = NULL;

  if (!_GetModelFilePath (&model_file) || !_GetOrangePngFilePath (&input_file))
    return NULL;

  g_autofree gchar *filter0 = g_strdup_printf (branch0, model_file);
  g_autofree gchar *filter1 = branch1 ? g_strdup_printf (branch1, model_file) : NULL;
  g_autofree gchar *pipeline = g_strdup_printf (
      "filesrc location=\"%s\" ! pngdec ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_transform mode=arithmetic option=typecast:float32,div:127.5,add:-1.0 ! tee name=t "
      "t. ! queue ! %s "
      "%s%s",
      input_file, filter0, filter1 ? "t. ! queue ! " : "", filter1 ? filter1 : "");

  GstElement *gstpipe = gst_parse_launch (pipeline, &err);
  g_clear_error (&err);
  return gstpipe;
}

/**
 * @brief Counts buffers whose classification is correct.
 */
static void
count_output (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  guint8 is_float = 1;
  check_output (element, buffer, &is_float);
  g_atomic_int_inc ((gint *) user_data);
}

#define ORANGE_FILTER "tensor_filter framework=onnxruntime model=\"%s\" ! tensor_sink name=sink0"
#define ORANGE_FILTER1 "tensor_filter framework=onnxruntime model=\"%s\" ! tensor_sink name=sink1"

/**
 * @brief Two filters of one model in a pipeline with a cache context run on one session.
 */
TEST (nnstreamerFilterOnnxRuntimeSessionCache, twoFiltersShareOneSession)
{
  NnsSharedSlot *slot = nns_shared_slot_new ();
  NnsOnnxruntimeSessionCacheStats stats;
  gint results = 0;

  GstElement *gstpipe = _NewOrangePipeline (ORANGE_FILTER, ORANGE_FILTER1);
  ASSERT_TRUE (gstpipe != nullptr);
  GstContext *context = _NewSessionCacheContext (slot, "lru");
  gst_element_set_context (gstpipe, context);
  gst_context_unref (context);

  for (const gchar *name : { "sink0", "sink1" }) {
    GstElement *sink = gst_bin_get_by_name (GST_BIN (gstpipe), name);
    ASSERT_TRUE (sink != nullptr);
    g_signal_connect (sink, "new-data", (GCallback) count_output, &results);
    gst_object_unref (sink);
  }

  _PlayToEos (gstpipe);
  EXPECT_EQ (results, 2);
  ASSERT_TRUE (_GetSessionCacheStats (slot, &stats));
  EXPECT_EQ (stats.sessions_created, 1U);
  EXPECT_EQ (stats.entries, 1U);
  EXPECT_EQ (stats.consumers, 2U);
  EXPECT_EQ (stats.hits, 1U);
  EXPECT_GT (stats.bytes, 0U);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  ASSERT_TRUE (_GetSessionCacheStats (slot, &stats));
  EXPECT_EQ (stats.consumers, 0U);
  EXPECT_EQ (stats.replicas, 1U);

  gst_object_unref (gstpipe);
  nns_shared_slot_unref (slot);
}

/**
 * @brief A filter with a different I/O contract gets its own session.
 */
TEST (nnstreamerFilterOnnxRuntimeSessionCache, differentIoContractGetsItsOwnSession)
{
  NnsSharedSlot *slot = nns_shared_slot_new ();
  NnsOnnxruntimeSessionCacheStats stats;

  GstElement *gstpipe = _NewOrangePipeline (ORANGE_FILTER,
      "tensor_filter framework=onnxruntime input=224:224:3:1 inputtype=float32 model=\"%s\" ! fakesink");
  ASSERT_TRUE (gstpipe != nullptr);
  GstContext *context = _NewSessionCacheContext (slot, "asap");
  gst_element_set_context (gstpipe, context);
  gst_context_unref (context);

  _PlayToEos (gstpipe);
  ASSERT_TRUE (_GetSessionCacheStats (slot, &stats));
  EXPECT_EQ (stats.sessions_created, 2U);
  EXPECT_EQ (stats.entries, 2U);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  ASSERT_TRUE (_GetSessionCacheStats (slot, &stats));
  EXPECT_EQ (stats.replicas, 0U);
  EXPECT_EQ (stats.entries, 0U);

  gst_object_unref (gstpipe);
  nns_shared_slot_unref (slot);
}

/**
 * @brief Pipelines alive at the same time share a session even with asap, as a pop swap does.
 */
TEST (nnstreamerFilterOnnxRuntimeSessionCache, overlappingPipelinesShareASession)
{
  NnsSharedSlot *slot = nns_shared_slot_new ();
  NnsOnnxruntimeSessionCacheStats stats;
  GstElement *pipelines[2];

  for (auto &pipeline : pipelines) {
    pipeline = _NewOrangePipeline (ORANGE_FILTER, NULL);
    ASSERT_TRUE (pipeline != nullptr);
    GstContext *context = _NewSessionCacheContext (slot, "asap");
    gst_element_set_context (pipeline, context);
    gst_context_unref (context);
    _PlayToEos (pipeline);
  }

  ASSERT_TRUE (_GetSessionCacheStats (slot, &stats));
  EXPECT_EQ (stats.sessions_created, 1U);
  EXPECT_EQ (stats.consumers, 2U);

  for (auto &pipeline : pipelines) {
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
    gst_object_unref (pipeline);
  }
  ASSERT_TRUE (_GetSessionCacheStats (slot, &stats));
  EXPECT_EQ (stats.replicas, 0U);
  EXPECT_EQ (stats.sessions_destroyed, 1U);
  nns_shared_slot_unref (slot);
}

/**
 * @brief lru keeps a session for the next pipeline; asap does not.
 */
TEST (nnstreamerFilterOnnxRuntimeSessionCache, lruRetainsAcrossPipelines)
{
  for (const gchar *strategy : { "lru", "asap" }) {
    NnsSharedSlot *slot = nns_shared_slot_new ();
    NnsOnnxruntimeSessionCacheStats stats;

    for (int run = 0; run < 2; run++) {
      GstElement *gstpipe = _NewOrangePipeline (ORANGE_FILTER, NULL);
      ASSERT_TRUE (gstpipe != nullptr);
      GstContext *context = _NewSessionCacheContext (slot, strategy);
      gst_element_set_context (gstpipe, context);
      gst_context_unref (context);
      _PlayToEos (gstpipe);
      EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
      gst_object_unref (gstpipe);
    }

    ASSERT_TRUE (_GetSessionCacheStats (slot, &stats));
    EXPECT_EQ (stats.sessions_created, g_str_equal (strategy, "lru") ? 1U : 2U) << strategy;
    nns_shared_slot_unref (slot);
  }
}

/**
 * @brief GPU tests run only where NNS_TEST_GPU=1 (a CUDA/TensorRT onnxruntime and a GPU).
 */
#define SKIP_WITHOUT_GPU()                                                   \
  do {                                                                       \
    if (g_strcmp0 (g_getenv ("NNS_TEST_GPU"), "1") != 0)                     \
      GTEST_SKIP () << "set NNS_TEST_GPU=1 to run GPU tests";               \
  } while (0)

#define GPU_FRAMES 40

/**
 * @brief GPU_FRAMES copies of the orange through one filter per branch, two branches.
 * accelerator goes before framework: only then does the V1 sub-plugin see accl_str, as with ep_infer.
 */
static GstElement *
_NewGpuPipeline (const gchar *filter_options, GstContext *context, gint *results)
{
  g_autofree gchar *model_file = NULL;
  g_autofree gchar *input_file = NULL;
  GError *err = NULL;

  if (!_GetModelFilePath (&model_file) || !_GetOrangePngFilePath (&input_file))
    return NULL;

  g_autofree gchar *pipeline = g_strdup_printf (
      "filesrc location=\"%s\" ! pngdec ! imagefreeze num-buffers=%d ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=25/1 ! tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_transform mode=arithmetic option=typecast:float32,div:127.5,add:-1.0 ! tee name=t "
      "t. ! queue ! tensor_filter accelerator=true:gpu framework=onnxruntime %s model=\"%s\" ! tensor_sink name=sink0 "
      "t. ! queue ! tensor_filter accelerator=true:gpu framework=onnxruntime %s model=\"%s\" ! tensor_sink name=sink1",
      input_file, GPU_FRAMES, filter_options, model_file, filter_options, model_file);

  GstElement *gstpipe = gst_parse_launch (pipeline, &err);
  g_clear_error (&err);
  if (!gstpipe)
    return NULL;
  gst_element_set_context (gstpipe, context);
  for (const gchar *name : { "sink0", "sink1" }) {
    GstElement *sink = gst_bin_get_by_name (GST_BIN (gstpipe), name);
    g_signal_connect (sink, "new-data", (GCallback) count_output, results);
    gst_object_unref (sink);
  }
  return gstpipe;
}

/**
 * @brief Outputs of one branch: every frame must equal the first.
 */
struct BranchRecord {
  GMutex lock;
  std::vector<guint8> first;
  gint frames = 0;
  gint mismatches = 0;

  BranchRecord ()
  {
    g_mutex_init (&lock);
  }
  ~BranchRecord ()
  {
    g_mutex_clear (&lock);
  }
};

/**
 * @brief Records a branch's output.
 */
static void
record_output (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  auto *record = static_cast<BranchRecord *> (user_data);
  GstMapInfo info;
  UNUSED (element);

  GstMemory *mem = gst_buffer_get_memory (buffer, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));
  g_mutex_lock (&record->lock);
  if (record->frames == 0) {
    record->first.assign (info.data, info.data + info.size);
  } else if (info.size != record->first.size ()
             || memcmp (info.data, record->first.data (), info.size) != 0) {
    record->mismatches++;
  }
  record->frames++;
  g_mutex_unlock (&record->lock);
  gst_memory_unmap (mem, &info);
  gst_memory_unref (mem);
}

/**
 * @brief GPU_FRAMES frames through conv_float_small.onnx, whose every node runs on CUDA,
 * in two branches fed with different inputs. Crossed buffers between consumers of one
 * session would show up as a branch disagreeing with itself or matching the other.
 */
static GstElement *
_NewGpuFloatPipeline (const gchar *filter_options, GstContext *context, BranchRecord *records)
{
  g_autofree gchar *input_file = NULL;
  GError *err = NULL;
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  g_autofree gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();
  g_autofree gchar *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "conv_float_small.onnx", NULL);

  if (!g_file_test (model_file, G_FILE_TEST_EXISTS) || !_GetOrangePngFilePath (&input_file))
    return NULL;

  g_autofree gchar *pipeline = g_strdup_printf (
      "filesrc location=\"%s\" ! pngdec ! imagefreeze num-buffers=%d ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=25/1 ! tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_transform mode=arithmetic option=typecast:float32,div:127.5 ! tee name=t "
      "t. ! queue ! tensor_transform mode=arithmetic option=add:-1.0 ! tensor_filter accelerator=true:gpu framework=onnxruntime %s model=\"%s\" ! tensor_sink name=sink0 "
      "t. ! queue ! tensor_transform mode=arithmetic option=add:-0.5 ! tensor_filter accelerator=true:gpu framework=onnxruntime %s model=\"%s\" ! tensor_sink name=sink1",
      input_file, GPU_FRAMES, filter_options, model_file, filter_options, model_file);

  GstElement *gstpipe = gst_parse_launch (pipeline, &err);
  g_clear_error (&err);
  if (!gstpipe)
    return NULL;
  gst_element_set_context (gstpipe, context);
  for (int i = 0; i < 2; i++) {
    g_autofree gchar *name = g_strdup_printf ("sink%d", i);
    GstElement *sink = gst_bin_get_by_name (GST_BIN (gstpipe), name);
    g_signal_connect (sink, "new-data", (GCallback) record_output, &records[i]);
    gst_object_unref (sink);
  }
  return gstpipe;
}

/**
 * @brief Each branch agrees with itself on every frame and differs from the other.
 */
static void
_ExpectConsistentBranches (BranchRecord *records)
{
  for (int i = 0; i < 2; i++) {
    EXPECT_EQ (records[i].frames, GPU_FRAMES) << "branch " << i;
    EXPECT_EQ (records[i].mismatches, 0) << "branch " << i;
  }
  EXPECT_NE (records[0].first, records[1].first);
}

/**
 * @brief Concurrent consumers of a CUDA graph session each run on an exclusive replica
 * with fixed buffers, so no result crosses over to the other consumer.
 */
TEST (nnstreamerFilterOnnxRuntimeSessionCacheGpu, cudaGraphConsumersRunOnExclusiveReplicas)
{
  SKIP_WITHOUT_GPU ();
  NnsSharedSlot *slot = nns_shared_slot_new ();
  NnsOnnxruntimeSessionCacheStats stats;
  BranchRecord records[2];

  GstContext *context = _NewSessionCacheContext (slot, "lru");
  GstElement *gstpipe = _NewGpuFloatPipeline ("", context, records);
  gst_context_unref (context);
  ASSERT_TRUE (gstpipe != nullptr);

  _PlayToEos (gstpipe, 120);
  _ExpectConsistentBranches (records);
  ASSERT_TRUE (_GetSessionCacheStats (slot, &stats));
  EXPECT_EQ (stats.poisons, 0U);
  EXPECT_EQ (stats.entries, 1U);
  EXPECT_GE (stats.replicas, 1U);
  EXPECT_LE (stats.replicas, 2U);
  EXPECT_GT (stats.bytes, 0U);
  g_message ("cuda graph: replicas=%" G_GUINT64_FORMAT " grows=%" G_GUINT64_FORMAT
             " bytes=%" G_GUINT64_FORMAT, (guint64) stats.replicas,
      (guint64) stats.grows, (guint64) stats.bytes);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (gstpipe);
  nns_shared_slot_unref (slot);
}

/**
 * @brief Without CUDA graph, consumers share one session and run concurrently.
 */
TEST (nnstreamerFilterOnnxRuntimeSessionCacheGpu, cudaWithoutGraphSharesOneReplica)
{
  SKIP_WITHOUT_GPU ();
  NnsSharedSlot *slot = nns_shared_slot_new ();
  NnsOnnxruntimeSessionCacheStats stats;
  BranchRecord records[2];

  GstContext *context = _NewSessionCacheContext (slot, "lru");
  GstElement *gstpipe = _NewGpuFloatPipeline ("custom=disable_cuda_graph", context, records);
  gst_context_unref (context);
  ASSERT_TRUE (gstpipe != nullptr);

  _PlayToEos (gstpipe, 120);
  _ExpectConsistentBranches (records);
  ASSERT_TRUE (_GetSessionCacheStats (slot, &stats));
  EXPECT_EQ (stats.replicas, 1U);
  EXPECT_EQ (stats.grows, 0U);
  EXPECT_EQ (stats.consumers, 2U);
  EXPECT_GT (stats.bytes, 0U);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (gstpipe);
  nns_shared_slot_unref (slot);
}

/**
 * @brief A model the CUDA EP cannot fully take rejects graph capture; the key is
 * poisoned and every consumer runs on the graph-off session.
 */
TEST (nnstreamerFilterOnnxRuntimeSessionCacheGpu, rejectedGraphCaptureFallsBackForEveryConsumer)
{
  SKIP_WITHOUT_GPU ();
  NnsSharedSlot *slot = nns_shared_slot_new ();
  NnsOnnxruntimeSessionCacheStats stats;
  gint results = 0;

  GstContext *context = _NewSessionCacheContext (slot, "lru");
  GstElement *gstpipe = _NewGpuPipeline ("", context, &results);
  gst_context_unref (context);
  ASSERT_TRUE (gstpipe != nullptr);

  _PlayToEos (gstpipe, 120);
  EXPECT_EQ (results, 2 * GPU_FRAMES);
  ASSERT_TRUE (_GetSessionCacheStats (slot, &stats));
  EXPECT_EQ (stats.poisons, 1U);
  EXPECT_EQ (stats.entries, 2U);
  EXPECT_EQ (stats.replicas, 1U);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (gstpipe);
  nns_shared_slot_unref (slot);
}

/**
 * @brief TensorRT sessions are exclusive and, with lru, reused by the next pipeline.
 */
TEST (nnstreamerFilterOnnxRuntimeSessionCacheGpu, tensorrtReplicasAreReusedAcrossPipelines)
{
  SKIP_WITHOUT_GPU ();
  NnsSharedSlot *slot = nns_shared_slot_new ();
  NnsOnnxruntimeSessionCacheStats stats;
  guint64 created_by_first = 0;

  for (int run = 0; run < 2; run++) {
    BranchRecord records[2];
    GstContext *context = _NewSessionCacheContext (slot, "lru");
    GstElement *gstpipe = _NewGpuFloatPipeline ("custom=enable_tensorrt", context, records);
    gst_context_unref (context);
    ASSERT_TRUE (gstpipe != nullptr);

    _PlayToEos (gstpipe, 600);
    _ExpectConsistentBranches (records);
    ASSERT_TRUE (_GetSessionCacheStats (slot, &stats));
    EXPECT_EQ (stats.poisons, 0U);
    EXPECT_LE (stats.replicas, 2U);
    if (run == 0)
      created_by_first = stats.sessions_created;
    else
      EXPECT_EQ (stats.sessions_created, created_by_first);

    EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
    gst_object_unref (gstpipe);
  }
  nns_shared_slot_unref (slot);
}

/**
 * @brief Main GTest
 */
int
main (int argc, char **argv)
{
  int result = -1;

  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);

  /* Force the binary to use dlog_print of untitest-util by calling it directly */
  ml_logd ("onnxruntime test starts w/ dummy backend.");

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
