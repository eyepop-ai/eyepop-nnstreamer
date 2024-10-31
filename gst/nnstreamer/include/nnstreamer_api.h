#ifndef __NNS_API_H__
#define __NNS_API_H__

#if (defined(_WIN32) || defined(__CYGWIN__)) && !defined(NNS_STATIC_COMPILATION)
#  define _NNS_EXPORT __declspec(dllexport)
#  define _NNS_IMPORT __declspec(dllimport)
#elif __GNUC__ >= 4
#  define _NNS_EXPORT __attribute__((visibility("default")))
#  define _NNS_IMPORT
#else
#  define _NNS_EXPORT
#  define _NNS_IMPORT
#endif

#define NNS_API_EXPORT _NNS_EXPORT extern
#define NNS_API_IMPORT _NNS_IMPORT extern

#endif
