/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    tensor_filter_onnxruntime_session_cache.hh
 * @brief   Shares onnxruntime sessions between tensor_filter instances.
 *
 * Consumers (tensor_filter instances) that resolve to the same SessionKey share
 * one cache Entry. An Entry holds a pool of replicas (sessions). A consumer
 * leases a replica per invoke:
 *  - an Unbounded replica serves any number of concurrent leases
 *    (EPs whose Run() is safe to call concurrently),
 *  - an Exclusive replica serves one lease at a time (EPs that serialize Run()
 *    or capture a CUDA graph), first come, first served; under contention the
 *    Entry grows another replica asynchronously, up to max_replicas_per_key.
 * A consumer that finds its primary Entry broken (poisoned) is redirected to a
 * fallback Entry, and so is every later consumer of that key.
 *
 * The cache core knows nothing about onnxruntime: replicas are built by an
 * injected factory and device memory is measured by an injected probe.
 */
#ifndef __TENSOR_FILTER_ONNXRUNTIME_SESSION_CACHE_HH__
#define __TENSOR_FILTER_ONNXRUNTIME_SESSION_CACHE_HH__

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nnstreamer
{
namespace tensor_filter_onnxruntime
{

enum class CacheStrategy { Asap, Lru };

struct CacheConfig {
  CacheStrategy strategy = CacheStrategy::Asap;
  uint64_t max_sessions = 0; /**< 0 = unlimited */
  uint64_t max_bytes = 0; /**< 0 = unlimited */
  unsigned max_replicas_per_key = 1;
};

/** @brief Identity of a shareable session, hashed once when a consumer starts. */
struct SessionKey {
  std::string canonical;
  size_t hash = 0;

  SessionKey () = default;
  explicit SessionKey (std::string canonical_);
  bool operator== (const SessionKey &other) const
  {
    return hash == other.hash && canonical == other.canonical;
  }
};

struct SessionKeyHash {
  size_t operator() (const SessionKey &key) const
  {
    return key.hash;
  }
};

enum class Concurrency { Unbounded, Exclusive };

/** @brief A session built by a ReplicaSpec factory; owned by the cache. */
class Replica
{
  public:
  virtual ~Replica () = default;
};

/** @brief How to build the replicas of one key. */
struct ReplicaSpec {
  SessionKey key;
  Concurrency concurrency = Concurrency::Unbounded;
  int device = -1; /**< device whose memory is measured around creation; -1 = host */
  /**
   * @brief Runs per replica and thread that hold the device lock. CUDA graph capture
   * (global capture mode) breaks if another thread creates or destroys a session, or
   * captures, meanwhile; the CUDA EP captures per thread, TensorRT after a warm-up run.
   */
  unsigned serialized_warmup_runs = 0;
  std::function<std::unique_ptr<Replica> ()> create; /**< may throw */
  std::function<uint64_t ()> host_bytes; /**< size estimate when no device probe applies; optional */
};

/** @brief Bytes in use on a device, or nullopt if unknown. */
using DeviceMemoryProbe = std::function<std::optional<uint64_t> (int device)>;

struct CacheStats {
  uint64_t entries = 0;
  uint64_t replicas = 0;
  uint64_t bytes = 0;
  uint64_t consumers = 0;
  uint64_t sessions_created = 0;
  uint64_t sessions_destroyed = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t evictions = 0;
  uint64_t overcommits = 0;
  uint64_t grows = 0;
  uint64_t grow_failures = 0;
  uint64_t poisons = 0;
};

class SessionCache;
class Entry;
struct ReplicaSlot;

/** @brief A replica leased for one invoke; returned on destruction. */
class Lease
{
  public:
  Lease () = default;
  Lease (Lease &&other) noexcept;
  Lease &operator= (Lease &&other) noexcept;
  Lease (const Lease &) = delete;
  Lease &operator= (const Lease &) = delete;
  ~Lease ();

  explicit operator bool () const
  {
    return slot_ != nullptr;
  }
  Replica *replica () const;
  /** @brief Whether the Entry had other consumers when this lease was handed out. */
  bool contended () const
  {
    return contended_;
  }
  /** @brief Runs fn; the first run of a replica is measured and added to its size. */
  void run (const std::function<void ()> &fn);
  void release ();

  private:
  friend class SessionCache;
  std::shared_ptr<SessionCache> cache_;
  std::shared_ptr<Entry> entry_;
  ReplicaSlot *slot_ = nullptr;
  bool first_use_ = false;
  bool serialize_ = false;
  bool contended_ = false;
};

/** @brief A consumer's reference to an Entry; keeps the Entry alive. */
class EntryRef
{
  public:
  EntryRef () = default;
  EntryRef (EntryRef &&other) noexcept;
  EntryRef &operator= (EntryRef &&other) noexcept;
  EntryRef (const EntryRef &) = delete;
  EntryRef &operator= (const EntryRef &) = delete;
  ~EntryRef ();

  explicit operator bool () const
  {
    return entry_ != nullptr;
  }
  /** @brief A replica with free capacity; follows a redirect of a poisoned Entry. May throw. */
  Lease lease ();
  /**
   * @brief Calls fn with some replica without leasing it, for reading what every replica of the
   * key shares (e.g. model I/O). fn must not run the replica. Builds one if there is none. May throw.
   */
  void inspect (const std::function<void (const Replica &)> &fn);
  /**
   * @brief Marks the Entry broken and moves this consumer (and every later one) to fallback.
   * Idle replicas of the broken Entry are destroyed; leased ones when returned.
   */
  void poison (const ReplicaSpec &fallback);
  const ReplicaSpec &spec () const;
  void reset ();

  private:
  friend class SessionCache;
  std::shared_ptr<SessionCache> cache_;
  std::shared_ptr<Entry> entry_;
};

class SessionCache : public std::enable_shared_from_this<SessionCache>
{
  public:
  static std::shared_ptr<SessionCache> create (
      const CacheConfig &config, DeviceMemoryProbe probe = nullptr);
  ~SessionCache ();

  /**
   * @brief A consumer's ref to the Entry of spec.key, with at least one replica built.
   * If building a replica throws and fallback is given, the key is poisoned and
   * the consumer lands on fallback. Otherwise the exception propagates.
   */
  EntryRef acquire (const ReplicaSpec &spec, const ReplicaSpec *fallback = nullptr);

  const CacheConfig &config () const
  {
    return config_;
  }
  CacheStats stats () const;

  private:
  friend class Lease;
  friend class EntryRef;

  SessionCache (const CacheConfig &config, DeviceMemoryProbe probe);

  std::shared_ptr<Entry> entry_for (const ReplicaSpec &spec); /**< cache_mu_ held */
  void add_consumer (const std::shared_ptr<Entry> &entry); /**< cache_mu_ held */
  void ensure_replica (const std::shared_ptr<Entry> &entry);
  std::unique_ptr<ReplicaSlot> build (const ReplicaSpec &spec);
  void grow_async (const std::shared_ptr<Entry> &entry);
  Lease lease (const std::shared_ptr<Entry> &entry, std::shared_ptr<Entry> &redirect);
  void return_lease (Lease &lease);
  void run_measured (Lease &lease, const std::function<void ()> &fn);
  bool inspect (const std::shared_ptr<Entry> &entry, const std::function<void (const Replica &)> &fn);
  void release_consumer (const std::shared_ptr<Entry> &entry);
  std::shared_ptr<Entry> follow (const std::shared_ptr<Entry> &from);
  std::shared_ptr<Entry> poison (const std::shared_ptr<Entry> &entry, const ReplicaSpec &fallback);
  void evict ();
  uint64_t tick ()
  {
    return ++clock_;
  }

  const CacheConfig config_;
  const DeviceMemoryProbe probe_;

  mutable std::mutex cache_mu_; /**< guards entries_; taken before any Entry mutex */
  std::unordered_map<SessionKey, std::shared_ptr<Entry>, SessionKeyHash> entries_;

  std::atomic<uint64_t> clock_{ 0 };
  std::atomic<uint64_t> sessions_created_{ 0 };
  std::atomic<uint64_t> sessions_destroyed_{ 0 };
  std::atomic<uint64_t> hits_{ 0 };
  std::atomic<uint64_t> misses_{ 0 };
  std::atomic<uint64_t> evictions_{ 0 };
  std::atomic<uint64_t> overcommits_{ 0 };
  std::atomic<uint64_t> grows_{ 0 };
  std::atomic<uint64_t> grow_failures_{ 0 };
  std::atomic<uint64_t> poisons_{ 0 };
};

} /* namespace tensor_filter_onnxruntime */
} /* namespace nnstreamer */

#endif /* __TENSOR_FILTER_ONNXRUNTIME_SESSION_CACHE_HH__ */
