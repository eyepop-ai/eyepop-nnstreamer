/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer python3 support
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file    nnstreamer_python3_helper.h
 * @date    21 May 2021
 * @brief   Common functions for various tensor_filters
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Gichan Jang <gichan2.jang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __NNS_PYTHON_HELPER_H__
#define __NNS_PYTHON_HELPER_H__

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <patchlevel.h>
#include <dlfcn.h>
#include <numpy/arrayobject.h>
#include <structmember.h>
#include <nnstreamer_api.h>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <stdexcept>
#include <string>

#ifdef __MACOS__
#define SO_EXT "dylib"
#else
#define SO_EXT "so.1.0"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define Py_ERRMSG(...)     \
  do {                     \
    if (PyErr_Occurred ()) \
      PyErr_Print ();      \
    PyErr_Clear();         \
    ml_loge (__VA_ARGS__); \
  } while (0);

#define Py_SAFEDECREF(o)   \
  do {                     \
    if (o) {               \
      Py_XDECREF (o);      \
      (o) = NULL;          \
    }                      \
  } while (0)

#if (PY_MAJOR_VERSION > 3) || ((PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION >= 7))
/* Since 3.7, PyEval_InitThreads does nothing and deprecated */
#define PyEval_InitThreads_IfGood()     do { } while (0)
#else
#define PyEval_InitThreads_IfGood()     do { PyEval_InitThreads(); } while (0)
#endif

NNS_API_EXPORT tensor_type getTensorType (NPY_TYPES npyType);
NNS_API_EXPORT NPY_TYPES getNumpyType (tensor_type tType);
NNS_API_EXPORT int loadScript (PyObject **core_obj, const gchar *module_name, const gchar *class_name);
NNS_API_EXPORT int openPythonLib (void **handle);
NNS_API_EXPORT int addToSysPath (const gchar *path);
NNS_API_EXPORT int parseTensorsInfo (PyObject *result, GstTensorsInfo *info);
NNS_API_EXPORT PyObject * PyTensorShape_New (PyObject * shape_cls, const GstTensorInfo *info);

/**
 * @brief Py_Initialize common wrapper for Python subplugins
 * @note This prevents a python-using subplugin finalizing another subplugin's python interpreter by sharing the reference counter.
 */
NNS_API_EXPORT void nnstreamer_python_init_refcnt ();

/**
 * @brief Py_Finalize common wrapper for Python subplugins
 * @note This prevents a python-using subplugin finalizing another subplugin's python interpreter by sharing the reference counter.
 */
NNS_API_EXPORT void nnstreamer_python_fini_refcnt ();

/**
 * @brief Check Py_Init status for python eval functions.
 * @return 0 if it's ready. negative error value if it's not ready.
 */
NNS_API_EXPORT int nnstreamer_python_status_check ();

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* __NNS_PYTHON_HELPER_H__ */
