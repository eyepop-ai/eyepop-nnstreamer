/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    nnstreamer_onnxruntime_session_cache.h
 * @brief   GstContext contract of the onnxruntime tensor_filter session cache.
 *
 * An application opts into sharing onnxruntime sessions between the
 * tensor_filter instances of a pipeline by setting a GstContext of type
 * NNS_ONNXRUNTIME_SESSION_CACHE_CONTEXT_TYPE on the pipeline (or on any bin
 * before it goes to READY). The context structure carries:
 *
 *  - "slot" (NnsSharedSlot, required): an empty slot, filled by the sub-plugin
 *  - "strategy" (string, optional): "asap" (default) or "lru"
 *  - "max-sessions" (guint64, optional): lru limit on sessions, 0 = unlimited
 *  - "max-bytes" (guint64, optional): lru limit on session memory, 0 = unlimited
 *  - "max-replicas-per-key" (guint, optional): default 1
 *
 * Without the context every tensor_filter keeps a private session.
 */
#ifndef __NNSTREAMER_ONNXRUNTIME_SESSION_CACHE_H__
#define __NNSTREAMER_ONNXRUNTIME_SESSION_CACHE_H__

#include <stdint.h>
#include "nnstreamer_shared_slot.h"

G_BEGIN_DECLS

#define NNS_ONNXRUNTIME_SESSION_CACHE_CONTEXT_TYPE "nnstreamer.onnxruntime.session-cache"
#define NNS_ONNXRUNTIME_SESSION_CACHE_FIELD_SLOT "slot"
#define NNS_ONNXRUNTIME_SESSION_CACHE_FIELD_STRATEGY "strategy"
#define NNS_ONNXRUNTIME_SESSION_CACHE_FIELD_MAX_SESSIONS "max-sessions"
#define NNS_ONNXRUNTIME_SESSION_CACHE_FIELD_MAX_BYTES "max-bytes"
#define NNS_ONNXRUNTIME_SESSION_CACHE_FIELD_MAX_REPLICAS_PER_KEY "max-replicas-per-key"

/** @brief Counters of one cache; the gauges (entries .. consumers) are current values. */
typedef struct
{
  uint64_t entries;
  uint64_t replicas;
  uint64_t bytes;
  uint64_t consumers;
  uint64_t sessions_created;
  uint64_t sessions_destroyed;
  uint64_t hits;
  uint64_t misses;
  uint64_t evictions;
  uint64_t overcommits;
  uint64_t grows;
  uint64_t grow_failures;
  uint64_t poisons;
} NnsOnnxruntimeSessionCacheStats;

/**
 * @brief Name of the function the onnxruntime sub-plugin exports for reading the stats of the
 * cache in a slot. Look it up with g_module_symbol(); it has the type
 * NnsOnnxruntimeSessionCacheGetStatsFunc and returns FALSE while the slot is still empty.
 */
#define NNS_ONNXRUNTIME_SESSION_CACHE_GET_STATS_SYMBOL "nnstreamer_onnxruntime_session_cache_get_stats"

typedef gboolean (*NnsOnnxruntimeSessionCacheGetStatsFunc) (
    NnsSharedSlot * slot, NnsOnnxruntimeSessionCacheStats * stats);

G_END_DECLS

#endif /* __NNSTREAMER_ONNXRUNTIME_SESSION_CACHE_H__ */
