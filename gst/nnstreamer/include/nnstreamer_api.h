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

// TODO: this is backwards and should be inverted
// Did this to avoid having to track down all the libraries that produce headers with NNS_API
// for the purposes of the prototype
#ifdef NNS_API_IMPORTS
#define NNS_API _NNS_IMPORT
#else
#define NNS_API _NNS_EXPORT
#endif

#endif
