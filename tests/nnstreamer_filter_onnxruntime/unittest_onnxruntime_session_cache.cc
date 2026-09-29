/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_onnxruntime_session_cache.cc
 * @brief   Unit tests for the onnxruntime session cache core, with fake sessions
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../../ext/nnstreamer/tensor_filter/tensor_filter_onnxruntime_session_cache.hh"

using namespace nnstreamer::tensor_filter_onnxruntime;
using namespace std::chrono_literals;

namespace
{

class FakeReplica : public Replica
{
  public:
  FakeReplica (int id_, std::atomic<int> *destroyed_in_use_) : id (id_), destroyed_in_use (destroyed_in_use_)
  {
  }
  ~FakeReplica () override
  {
    if (in_use > 0)
      (*destroyed_in_use)++;
  }
  const int id;
  std::atomic<int> in_use{ 0 };
  std::atomic<int> *destroyed_in_use;
};

/** @brief Builds fake replicas; each one takes bytes_per_replica of fake device memory. */
struct FakeFactory {
  std::atomic<int> created{ 0 };
  std::atomic<int> fail_from{ -1 }; /**< creations with index >= fail_from throw */
  std::atomic<uint64_t> device_used{ 0 };
  std::atomic<int> destroyed_in_use{ 0 }; /**< replicas destroyed while a lease used them */
  uint64_t bytes_per_replica = 100;
  std::chrono::milliseconds delay{ 0 };

  ReplicaSpec spec (const std::string &key, Concurrency concurrency = Concurrency::Unbounded, int device = -1)
  {
    ReplicaSpec spec;
    spec.key = SessionKey (key);
    spec.concurrency = concurrency;
    spec.device = device;
    spec.create = [this] () -> std::unique_ptr<Replica> {
      if (delay.count ())
        std::this_thread::sleep_for (delay);
      int index = created++;
      if (fail_from >= 0 && index >= fail_from)
        throw std::runtime_error ("fake creation failure");
      device_used += bytes_per_replica;
      return std::make_unique<FakeReplica> (index, &destroyed_in_use);
    };
    spec.host_bytes = [this] () { return bytes_per_replica; };
    return spec;
  }

  DeviceMemoryProbe probe ()
  {
    return [this] (int) -> std::optional<uint64_t> { return device_used.load (); };
  }
};

int
replica_id (const Lease &lease)
{
  return static_cast<FakeReplica *> (lease.replica ())->id;
}

/** @brief Isolated device locks for one test's cache. */
std::shared_ptr<DeviceLocks>
locks ()
{
  return std::make_shared<DeviceLocks> ();
}

/** @brief Waits for async grow threads, which must not outlive the test locals they call. */
bool
wait_for_grows (const std::shared_ptr<SessionCache> &cache)
{
  for (int i = 0; i < 300; i++) {
    if (cache->stats ().grows_in_flight == 0)
      return true;
    std::this_thread::sleep_for (10ms);
  }
  return false;
}

CacheConfig
lru (uint64_t max_sessions = 0, uint64_t max_bytes = 0, unsigned max_replicas = 2)
{
  CacheConfig config;
  config.strategy = CacheStrategy::Lru;
  config.max_sessions = max_sessions;
  config.max_bytes = max_bytes;
  config.max_replicas_per_key = max_replicas;
  return config;
}

CacheConfig
asap (unsigned max_replicas)
{
  CacheConfig config;
  config.max_replicas_per_key = max_replicas;
  return config;
}

} /* namespace */

TEST (onnxruntimeSessionCache, defaultIsOneReplicaPerKey)
{
  EXPECT_EQ (CacheConfig{}.max_replicas_per_key, 1u);
}

TEST (onnxruntimeSessionCache, keyIsHashedOnce)
{
  SessionKey a ("model|cpu|0");
  SessionKey b ("model|cpu|0");
  SessionKey c ("model|cuda|0");
  EXPECT_EQ (a.hash, b.hash);
  EXPECT_TRUE (a == b);
  EXPECT_FALSE (a == c);
}

TEST (onnxruntimeSessionCache, consumersOfOneKeyShareOneReplica)
{
  FakeFactory factory;
  auto cache = SessionCache::create (CacheConfig{}, locks ());
  auto spec = factory.spec ("a");

  EntryRef first = cache->acquire (spec);
  EntryRef second = cache->acquire (spec);

  EXPECT_EQ (factory.created, 1);
  CacheStats stats = cache->stats ();
  EXPECT_EQ (stats.entries, 1u);
  EXPECT_EQ (stats.replicas, 1u);
  EXPECT_EQ (stats.consumers, 2u);
  EXPECT_EQ (stats.misses, 1u);
  EXPECT_EQ (stats.hits, 1u);
}

TEST (onnxruntimeSessionCache, differentKeysGetDifferentEntries)
{
  FakeFactory factory;
  auto cache = SessionCache::create (CacheConfig{}, locks ());

  EntryRef a = cache->acquire (factory.spec ("a"));
  EntryRef b = cache->acquire (factory.spec ("b"));

  EXPECT_EQ (factory.created, 2);
  EXPECT_EQ (cache->stats ().entries, 2u);
}

TEST (onnxruntimeSessionCache, asapDestroysWithTheLastConsumer)
{
  FakeFactory factory;
  auto cache = SessionCache::create (CacheConfig{}, locks ());
  auto spec = factory.spec ("a");

  EntryRef first = cache->acquire (spec);
  EntryRef second = cache->acquire (spec);
  first.reset ();
  EXPECT_EQ (cache->stats ().replicas, 1u);

  second.reset ();
  CacheStats stats = cache->stats ();
  EXPECT_EQ (stats.replicas, 0u);
  EXPECT_EQ (stats.entries, 0u);
  EXPECT_EQ (stats.sessions_destroyed, 1u);

  EntryRef again = cache->acquire (spec);
  EXPECT_EQ (factory.created, 2);
}

TEST (onnxruntimeSessionCache, lruRetainsUnreferencedReplicas)
{
  FakeFactory factory;
  auto cache = SessionCache::create (lru (), locks ());
  auto spec = factory.spec ("a");

  cache->acquire (spec).reset ();
  EXPECT_EQ (cache->stats ().replicas, 1u);

  EntryRef again = cache->acquire (spec);
  EXPECT_EQ (factory.created, 1);
  EXPECT_EQ (cache->stats ().hits, 1u);
}

TEST (onnxruntimeSessionCache, lruMaxSessionsEvictsTheOldestUnreferenced)
{
  FakeFactory factory;
  auto cache = SessionCache::create (lru (2), locks ());

  cache->acquire (factory.spec ("a")).reset ();
  cache->acquire (factory.spec ("b")).reset ();
  cache->acquire (factory.spec ("c")).reset ();

  CacheStats stats = cache->stats ();
  EXPECT_EQ (stats.replicas, 2u);
  EXPECT_EQ (stats.evictions, 1u);

  cache->acquire (factory.spec ("b")).reset ();
  cache->acquire (factory.spec ("c")).reset ();
  EXPECT_EQ (factory.created, 3);
  cache->acquire (factory.spec ("a")).reset ();
  EXPECT_EQ (factory.created, 4);
}

TEST (onnxruntimeSessionCache, lruMaxBytesUsesTheMeasuredDeviceSize)
{
  FakeFactory factory;
  auto cache = SessionCache::create (lru (0, 250), locks (), factory.probe ());

  cache->acquire (factory.spec ("a", Concurrency::Unbounded, 0)).reset ();
  cache->acquire (factory.spec ("b", Concurrency::Unbounded, 0)).reset ();
  EXPECT_EQ (cache->stats ().bytes, 200u);

  cache->acquire (factory.spec ("c", Concurrency::Unbounded, 0)).reset ();
  CacheStats stats = cache->stats ();
  EXPECT_EQ (stats.bytes, 200u);
  EXPECT_EQ (stats.replicas, 2u);
  EXPECT_EQ (stats.evictions, 1u);
}

TEST (onnxruntimeSessionCache, limitsCombine)
{
  FakeFactory factory;
  auto cache = SessionCache::create (lru (10, 150), locks (), factory.probe ());

  cache->acquire (factory.spec ("a", Concurrency::Unbounded, 0)).reset ();
  cache->acquire (factory.spec ("b", Concurrency::Unbounded, 0)).reset ();
  EXPECT_EQ (cache->stats ().replicas, 1u);
}

TEST (onnxruntimeSessionCache, referencedReplicasAreOvercommittedNotEvicted)
{
  FakeFactory factory;
  auto cache = SessionCache::create (lru (1), locks ());

  EntryRef a = cache->acquire (factory.spec ("a"));
  EntryRef b = cache->acquire (factory.spec ("b"));

  CacheStats stats = cache->stats ();
  EXPECT_EQ (stats.replicas, 2u);
  EXPECT_EQ (stats.evictions, 0u);
  EXPECT_GE (stats.overcommits, 1u);

  a.reset ();
  stats = cache->stats ();
  EXPECT_EQ (stats.replicas, 1u);
  EXPECT_EQ (stats.evictions, 1u);
}

TEST (onnxruntimeSessionCache, unboundedReplicaServesConcurrentLeases)
{
  FakeFactory factory;
  auto cache = SessionCache::create (CacheConfig{}, locks ());
  EntryRef a = cache->acquire (factory.spec ("a"));
  EntryRef b = cache->acquire (factory.spec ("a"));

  Lease la = a.lease ();
  Lease lb = b.lease ();
  EXPECT_EQ (la.replica (), lb.replica ());
  EXPECT_EQ (factory.created, 1);
  EXPECT_EQ (cache->stats ().grows, 0u);
}

TEST (onnxruntimeSessionCache, exclusiveGrowsAsynchronouslyUnderContention)
{
  FakeFactory factory;
  factory.delay = 50ms;
  auto cache = SessionCache::create (asap (2), locks ());
  auto spec = factory.spec ("a", Concurrency::Exclusive);
  EntryRef a = cache->acquire (spec);
  EntryRef b = cache->acquire (spec);

  Lease la = a.lease ();
  auto contended = std::async (std::launch::async, [&b] () {
    Lease lb = b.lease ();
    return replica_id (lb);
  });

  EXPECT_EQ (contended.wait_for (2s), std::future_status::ready);
  EXPECT_EQ (contended.get (), 1);
  EXPECT_EQ (replica_id (la), 0);
  CacheStats stats = cache->stats ();
  EXPECT_EQ (stats.grows, 1u);
  EXPECT_EQ (stats.replicas, 2u);
  EXPECT_TRUE (wait_for_grows (cache));
}

TEST (onnxruntimeSessionCache, contendedLeaseTakesAReturnedReplicaBeforeTheGrowFinishes)
{
  FakeFactory factory;
  auto cache = SessionCache::create (asap (2), locks ());
  auto spec = factory.spec ("a", Concurrency::Exclusive);
  EntryRef a = cache->acquire (spec);
  EntryRef b = cache->acquire (spec);
  factory.delay = 500ms;

  Lease la = a.lease ();
  auto contended = std::async (std::launch::async, [&b] () {
    Lease lb = b.lease ();
    return replica_id (lb);
  });
  std::this_thread::sleep_for (50ms);
  la.release ();

  EXPECT_EQ (contended.wait_for (200ms), std::future_status::ready);
  EXPECT_EQ (contended.get (), 0);
  EXPECT_TRUE (wait_for_grows (cache));
  EXPECT_EQ (cache->stats ().replicas, 2u);
}

TEST (onnxruntimeSessionCache, exclusiveGrowthIsCapped)
{
  FakeFactory factory;
  auto cache = SessionCache::create (lru (0, 0, 1), locks ());
  auto spec = factory.spec ("a", Concurrency::Exclusive);
  EntryRef a = cache->acquire (spec);
  EntryRef b = cache->acquire (spec);

  Lease la = a.lease ();
  auto contended = std::async (std::launch::async, [&b] () {
    Lease lb = b.lease ();
    return replica_id (lb);
  });
  EXPECT_EQ (contended.wait_for (100ms), std::future_status::timeout);
  la.release ();
  EXPECT_EQ (contended.get (), 0);
  EXPECT_EQ (factory.created, 1);
  EXPECT_EQ (cache->stats ().grows, 0u);
}

TEST (onnxruntimeSessionCache, failedGrowStopsGrowingThatKey)
{
  FakeFactory factory;
  auto cache = SessionCache::create (asap (2), locks ());
  auto spec = factory.spec ("a", Concurrency::Exclusive);
  EntryRef a = cache->acquire (spec);
  EntryRef b = cache->acquire (spec);
  factory.fail_from = 1;

  Lease la = a.lease ();
  auto contended = std::async (std::launch::async, [&b] () {
    Lease lb = b.lease ();
    return replica_id (lb);
  });
  std::this_thread::sleep_for (100ms);
  EXPECT_EQ (cache->stats ().grow_failures, 1u);
  la.release ();
  EXPECT_EQ (contended.get (), 0);

  la = a.lease ();
  auto again = std::async (std::launch::async, [&b] () {
    Lease lb = b.lease ();
    return replica_id (lb);
  });
  EXPECT_EQ (again.wait_for (100ms), std::future_status::timeout);
  la.release ();
  again.get ();
  EXPECT_EQ (cache->stats ().grows, 1u);
  EXPECT_TRUE (wait_for_grows (cache));
}

TEST (onnxruntimeSessionCache, poisonRedirectsEveryConsumerOfTheKey)
{
  FakeFactory factory;
  auto cache = SessionCache::create (lru (), locks ());
  auto primary = factory.spec ("a|graph");
  auto fallback = factory.spec ("a|nograph");
  EntryRef a = cache->acquire (primary, &fallback);
  EntryRef b = cache->acquire (primary, &fallback);

  a.poison (fallback);
  EXPECT_TRUE (a.spec ().key == fallback.key);
  EXPECT_TRUE (b.spec ().key == primary.key);

  Lease lb = b.lease ();
  EXPECT_TRUE (b.spec ().key == fallback.key);
  lb.release ();

  EntryRef late = cache->acquire (primary, &fallback);
  EXPECT_TRUE (late.spec ().key == fallback.key);

  CacheStats stats = cache->stats ();
  EXPECT_EQ (stats.poisons, 1u);
  EXPECT_EQ (factory.created, 2);
  EXPECT_EQ (stats.replicas, 1u);
  EXPECT_EQ (stats.consumers, 3u);
}

TEST (onnxruntimeSessionCache, poisonedReplicaInUseIsDestroyedWhenReturned)
{
  FakeFactory factory;
  auto cache = SessionCache::create (CacheConfig{}, locks ());
  auto primary = factory.spec ("a|graph", Concurrency::Exclusive);
  auto fallback = factory.spec ("a|nograph", Concurrency::Unbounded);
  EntryRef a = cache->acquire (primary);
  EntryRef b = cache->acquire (primary);

  Lease la = a.lease ();
  b.poison (fallback);
  EXPECT_EQ (cache->stats ().sessions_destroyed, 0u);
  la.release ();
  EXPECT_EQ (cache->stats ().sessions_destroyed, 1u);

  Lease again = a.lease ();
  EXPECT_TRUE (a.spec ().key == fallback.key);
}

TEST (onnxruntimeSessionCache, creationFailureFallsBack)
{
  FakeFactory factory;
  auto cache = SessionCache::create (CacheConfig{}, locks ());
  auto primary = factory.spec ("a|graph");
  primary.create = [] () -> std::unique_ptr<Replica> { throw std::runtime_error ("no graph"); };
  auto fallback = factory.spec ("a|nograph");

  EntryRef a = cache->acquire (primary, &fallback);
  EXPECT_TRUE (a.spec ().key == fallback.key);
  EntryRef b = cache->acquire (primary, &fallback);
  EXPECT_TRUE (b.spec ().key == fallback.key);
  EXPECT_EQ (factory.created, 1);
  EXPECT_EQ (cache->stats ().poisons, 1u);
}

TEST (onnxruntimeSessionCache, creationFailureWithoutFallbackThrowsAndLeavesNothing)
{
  FakeFactory factory;
  factory.fail_from = 0;
  auto cache = SessionCache::create (CacheConfig{}, locks ());

  EXPECT_THROW (cache->acquire (factory.spec ("a")), std::runtime_error);
  CacheStats stats = cache->stats ();
  EXPECT_EQ (stats.entries, 0u);
  EXPECT_EQ (stats.consumers, 0u);
}

TEST (onnxruntimeSessionCache, firstRunIsMeasured)
{
  FakeFactory factory;
  auto cache = SessionCache::create (lru (), locks (), factory.probe ());
  EntryRef a = cache->acquire (factory.spec ("a", Concurrency::Exclusive, 0));
  EXPECT_EQ (cache->stats ().bytes, 100u);

  Lease first = a.lease ();
  first.run ([&factory] () { factory.device_used += 40; });
  first.release ();
  EXPECT_EQ (cache->stats ().bytes, 140u);

  Lease second = a.lease ();
  second.run ([&factory] () { factory.device_used += 40; });
  EXPECT_EQ (cache->stats ().bytes, 140u);
}

TEST (onnxruntimeSessionCache, exclusiveReplicasAreNeverSharedUnderStress)
{
  FakeFactory factory;
  factory.delay = 5ms;
  auto cache = SessionCache::create (lru (0, 0, 3), locks ());
  auto spec = factory.spec ("a", Concurrency::Exclusive);

  std::atomic<int> violations{ 0 };
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; t++) {
    threads.emplace_back ([&] () {
      EntryRef ref = cache->acquire (spec);
      for (int i = 0; i < 200; i++) {
        Lease lease = ref.lease ();
        auto *replica = static_cast<FakeReplica *> (lease.replica ());
        if (replica->in_use.fetch_add (1) != 0)
          violations++;
        std::this_thread::yield ();
        replica->in_use.fetch_sub (1);
      }
    });
  }
  for (auto &thread : threads)
    thread.join ();

  EXPECT_EQ (violations, 0);
  CacheStats stats = cache->stats ();
  EXPECT_LE (stats.replicas, 3u);
  EXPECT_EQ (stats.consumers, 0u);
  EXPECT_TRUE (wait_for_grows (cache));
}

TEST (onnxruntimeSessionCache, warmupRunsNeverOverlapCreationOrEachOther)
{
  FakeFactory factory;
  std::atomic<int> critical{ 0 };
  std::atomic<int> overlaps{ 0 };
  std::atomic<int> serialized_runs{ 0 };
  auto enter = [&] () {
    if (critical.fetch_add (1) != 0)
      overlaps++;
    std::this_thread::sleep_for (2ms);
    critical.fetch_sub (1);
  };

  auto cache = SessionCache::create (lru (0, 0, 4), locks (), factory.probe ());
  auto spec = factory.spec ("graph", Concurrency::Exclusive, 0);
  spec.serialized_warmup_runs = 2;
  auto create = spec.create;
  spec.create = [&, create] () {
    enter ();
    return create ();
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < 6; t++) {
    threads.emplace_back ([&] () {
      EntryRef ref = cache->acquire (spec);
      for (int i = 0; i < 20; i++) {
        Lease lease = ref.lease ();
        bool warmup = i < 2;
        lease.run ([&] () {
          if (warmup) {
            serialized_runs++;
            enter ();
          }
        });
      }
    });
  }
  for (auto &thread : threads)
    thread.join ();
  EXPECT_TRUE (wait_for_grows (cache));

  EXPECT_EQ (overlaps, 0);
  EXPECT_GE (serialized_runs, 12);
}

TEST (onnxruntimeSessionCache, threadGetsBackTheReplicaItWarmedUp)
{
  FakeFactory factory;
  auto cache = SessionCache::create (lru (0, 0, 2), locks ());
  auto spec = factory.spec ("graph", Concurrency::Exclusive);
  spec.serialized_warmup_runs = 1;
  EntryRef a = cache->acquire (spec);
  EntryRef b = cache->acquire (spec);

  Lease la = a.lease ();
  int mine = replica_id (la);
  auto other = std::async (std::launch::async, [&b] () {
    Lease lb = b.lease ();
    return replica_id (lb);
  });
  int theirs = other.get ();
  EXPECT_NE (mine, theirs);
  la.release ();

  /* the other thread's replica was used last, but this thread warmed up its own */
  for (int i = 0; i < 5; i++) {
    Lease again = a.lease ();
    EXPECT_EQ (replica_id (again), mine);
  }
}

TEST (onnxruntimeSessionCache, exclusiveLeasesAreFirstComeFirstServed)
{
  FakeFactory factory;
  auto cache = SessionCache::create (CacheConfig{}, locks ());
  auto spec = factory.spec ("a", Concurrency::Exclusive);
  EntryRef a = cache->acquire (spec);
  EntryRef b = cache->acquire (spec);
  EntryRef c = cache->acquire (spec);

  std::mutex order_mu;
  std::vector<char> order;
  auto record = [&] (char who) {
    std::lock_guard<std::mutex> lock (order_mu);
    order.push_back (who);
  };

  Lease la = a.lease ();
  auto tb = std::async (std::launch::async, [&] () {
    Lease lb = b.lease ();
    record ('b');
  });
  std::this_thread::sleep_for (50ms);
  auto tc = std::async (std::launch::async, [&] () {
    Lease lc = c.lease ();
    record ('c');
  });
  std::this_thread::sleep_for (50ms);

  /* a returns the replica while b and c wait, and asks again at once: it goes last */
  la.release ();
  la = a.lease ();
  record ('a');
  la.release ();
  tb.get ();
  tc.get ();

  ASSERT_EQ (order.size (), 3u);
  EXPECT_EQ (order[0], 'b');
  EXPECT_EQ (order[1], 'c');
  EXPECT_EQ (order[2], 'a');
  EXPECT_EQ (factory.created, 1);
}

TEST (onnxruntimeSessionCache, leaseTellsWhetherTheEntryIsShared)
{
  FakeFactory factory;
  auto cache = SessionCache::create (CacheConfig{}, locks ());
  auto spec = factory.spec ("a", Concurrency::Exclusive);
  EntryRef a = cache->acquire (spec);
  EXPECT_FALSE (a.lease ().contended ());
  EntryRef b = cache->acquire (spec);
  EXPECT_TRUE (a.lease ().contended ());
  b.reset ();
  EXPECT_FALSE (a.lease ().contended ());
}

TEST (onnxruntimeSessionCache, evictionNeverDestroysALeasedReplica)
{
  FakeFactory factory;
  auto cache = SessionCache::create (lru (1, 0, 4), locks ());
  auto spec = factory.spec ("busy", Concurrency::Exclusive);
  std::atomic<bool> stop{ false };

  /* consumers of one key keep up to 4 replicas, above the limit of 1, so every
   * eviction pass targets their idle surplus while they lease */
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; t++) {
    threads.emplace_back ([&] () {
      EntryRef ref = cache->acquire (spec);
      while (!stop) {
        Lease lease = ref.lease ();
        auto *replica = static_cast<FakeReplica *> (lease.replica ());
        replica->in_use++;
        std::this_thread::yield ();
        replica->in_use--;
      }
    });
  }
  /* churn another key to run eviction passes */
  for (int i = 0; i < 300; i++)
    cache->acquire (factory.spec ("churn")).reset ();
  stop = true;
  for (auto &thread : threads)
    thread.join ();
  EXPECT_TRUE (wait_for_grows (cache));

  EXPECT_EQ (factory.destroyed_in_use, 0);
  EXPECT_GT (cache->stats ().evictions, 0u);
}

TEST (onnxruntimeSessionCache, redirectTargetOutlivesItsLastConsumer)
{
  FakeFactory factory;
  auto cache = SessionCache::create (CacheConfig{}, locks ());
  auto primary = factory.spec ("a|graph");
  auto fallback = factory.spec ("a|nograph");

  EntryRef first = cache->acquire (primary, &fallback);
  first.poison (fallback);
  first.lease ().release ();
  first.reset ();

  /* asap destroyed the fallback's replica; the poisoned key must still reach the same entry */
  EntryRef again = cache->acquire (primary, &fallback);
  EXPECT_TRUE (again.spec ().key == fallback.key);
  EXPECT_EQ (cache->stats ().replicas, 1u);

  EntryRef direct = cache->acquire (fallback);
  int created = factory.created;
  EXPECT_EQ (cache->stats ().replicas, 1u);
  EXPECT_EQ (cache->stats ().consumers, 2u);
  EXPECT_EQ (factory.created, created);
}

TEST (onnxruntimeSessionCache, cachesWithSeparateDeviceLocksDoNotBlockEachOther)
{
  FakeFactory slow, fast;
  slow.delay = 300ms;
  auto slow_cache = SessionCache::create (CacheConfig{}, locks (), slow.probe ());
  auto fast_cache = SessionCache::create (CacheConfig{}, locks (), fast.probe ());

  auto building = std::async (std::launch::async, [&] () {
    return slow_cache->acquire (slow.spec ("slow", Concurrency::Exclusive, 0));
  });
  std::this_thread::sleep_for (50ms);
  auto start = std::chrono::steady_clock::now ();
  EntryRef quick = fast_cache->acquire (fast.spec ("fast", Concurrency::Exclusive, 0));
  EXPECT_LT (std::chrono::steady_clock::now () - start, 200ms);
  building.get ();
}

int
main (int argc, char **argv)
{
  testing::InitGoogleTest (&argc, argv);
  return RUN_ALL_TESTS ();
}
