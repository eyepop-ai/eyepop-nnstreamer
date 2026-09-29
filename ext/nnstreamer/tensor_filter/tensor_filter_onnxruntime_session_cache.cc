/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    tensor_filter_onnxruntime_session_cache.cc
 * @brief   Shares onnxruntime sessions between tensor_filter instances.
 *
 * Lock order: cache_mu_ before any Entry::mu. A device lock (DeviceLocks) is
 * never held together with either; it wraps creating and destroying device
 * replicas and their first runs.
 */

#include "tensor_filter_onnxruntime_session_cache.hh"

#include <algorithm>
#include <deque>
#include <unordered_set>
#include <stdexcept>
#include <system_error>
#include <thread>

#include <glib.h>

#ifdef G_LOG_DOMAIN
#undef G_LOG_DOMAIN
#endif
#define G_LOG_DOMAIN "eyepop-ai"

namespace nnstreamer
{
namespace tensor_filter_onnxruntime
{

struct ReplicaSlot {
  std::unique_ptr<Replica> replica;
  int device = -1;
  unsigned leases = 0;
  uint64_t last_used = 0;
  uint64_t bytes = 0;
  bool used = false; /**< a lease was handed out; the first one is measured */
  std::vector<std::pair<std::thread::id, unsigned>> runs_by_thread; /**< only with serialized warm-up */

  unsigned *runs_of (std::thread::id thread)
  {
    for (auto &runs : runs_by_thread)
      if (runs.first == thread)
        return &runs.second;
    return nullptr;
  }
};

/** @brief Bytes a device gained around fn, if probe could measure both ends. */
static bool
measure_growth (const DeviceMemoryProbe &probe, int device, const std::function<void ()> &fn, uint64_t &grown)
{
  uint64_t before = 0, after = 0;
  bool known = false;
  if (probe) {
    if (auto used = probe (device)) {
      before = *used;
      known = true;
    }
  }
  fn ();
  if (known) {
    auto used = probe (device);
    known = used.has_value ();
    after = known ? *used : 0;
  }
  grown = known && after > before ? after - before : 0;
  return known;
}

/** @brief Destroys slots; device replicas under their device lock. */
void
SessionCache::destroy_slots (std::vector<std::unique_ptr<ReplicaSlot>> &slots)
{
  for (auto &slot : slots) {
    if (slot && slot->device >= 0) {
      std::lock_guard<std::mutex> lock (device_locks_->of (slot->device));
      slot.reset ();
    }
  }
  slots.clear ();
}

class Entry
{
  public:
  explicit Entry (const ReplicaSpec &spec_) : spec (spec_)
  {
  }

  const ReplicaSpec spec;
  std::mutex mu;
  std::condition_variable cv; /**< a replica was returned, built or the Entry redirected */
  std::vector<std::unique_ptr<ReplicaSlot>> replicas;
  unsigned consumers = 0;
  unsigned building = 0;
  bool grow_failed = false; /**< an async grow failed; stop growing this Entry */
  bool poisoned = false;
  std::shared_ptr<Entry> redirect; /**< set with poisoned; written under cache_mu_ and mu */
  std::deque<uint64_t> waiting; /**< tickets of consumers waiting for an exclusive replica, in arrival order */
  uint64_t next_ticket = 0;

  std::unique_ptr<ReplicaSlot> take (ReplicaSlot *slot)
  {
    auto it = std::find_if (replicas.begin (), replicas.end (),
        [slot] (const std::unique_ptr<ReplicaSlot> &s) { return s.get () == slot; });
    if (it == replicas.end ())
      return nullptr;
    std::unique_ptr<ReplicaSlot> taken = std::move (*it);
    replicas.erase (it);
    return taken;
  }

  void take_idle (std::vector<std::unique_ptr<ReplicaSlot>> &into)
  {
    auto idle = std::stable_partition (replicas.begin (), replicas.end (),
        [] (const std::unique_ptr<ReplicaSlot> &s) { return s->leases > 0; });
    for (auto it = idle; it != replicas.end (); ++it)
      into.push_back (std::move (*it));
    replicas.erase (idle, replicas.end ());
  }
};

SessionKey::SessionKey (std::string canonical_)
    : canonical (std::move (canonical_)), hash (std::hash<std::string>{}(canonical))
{
}

/* Lease */

Lease::Lease (Lease &&other) noexcept
    : cache_ (std::move (other.cache_)), entry_ (std::move (other.entry_)),
      slot_ (other.slot_), first_use_ (other.first_use_), serialize_ (other.serialize_),
      contended_ (other.contended_)
{
  other.slot_ = nullptr;
}

Lease &
Lease::operator= (Lease &&other) noexcept
{
  if (this != &other) {
    release ();
    cache_ = std::move (other.cache_);
    entry_ = std::move (other.entry_);
    slot_ = other.slot_;
    first_use_ = other.first_use_;
    serialize_ = other.serialize_;
    contended_ = other.contended_;
    other.slot_ = nullptr;
  }
  return *this;
}

Lease::~Lease ()
{
  release ();
}

Replica *
Lease::replica () const
{
  return slot_ ? slot_->replica.get () : nullptr;
}

void
Lease::run (const std::function<void ()> &fn)
{
  if (!slot_)
    throw std::logic_error ("run on an empty lease");
  cache_->run_measured (*this, fn);
}

void
Lease::release ()
{
  if (slot_)
    cache_->return_lease (*this);
  slot_ = nullptr;
  entry_.reset ();
  cache_.reset ();
}

/* EntryRef */

EntryRef::EntryRef (EntryRef &&other) noexcept
    : cache_ (std::move (other.cache_)), entry_ (std::move (other.entry_))
{
}

EntryRef &
EntryRef::operator= (EntryRef &&other) noexcept
{
  if (this != &other) {
    reset ();
    cache_ = std::move (other.cache_);
    entry_ = std::move (other.entry_);
  }
  return *this;
}

EntryRef::~EntryRef ()
{
  reset ();
}

Lease
EntryRef::lease ()
{
  if (!entry_)
    throw std::logic_error ("lease on an empty entry ref");
  for (;;) {
    std::shared_ptr<Entry> redirect;
    Lease lease = cache_->lease (entry_, redirect);
    if (lease)
      return lease;
    entry_ = cache_->follow (entry_);
  }
}

void
EntryRef::inspect (const std::function<void (const Replica &)> &fn)
{
  if (!entry_)
    throw std::logic_error ("inspect on an empty entry ref");
  while (!cache_->inspect (entry_, fn))
    entry_ = cache_->follow (entry_);
}

void
EntryRef::poison (const ReplicaSpec &fallback)
{
  if (!entry_)
    throw std::logic_error ("poison on an empty entry ref");
  entry_ = cache_->poison (entry_, fallback);
}

const ReplicaSpec &
EntryRef::spec () const
{
  if (!entry_)
    throw std::logic_error ("spec of an empty entry ref");
  return entry_->spec;
}

void
EntryRef::reset ()
{
  if (entry_)
    cache_->release_consumer (entry_);
  entry_.reset ();
  cache_.reset ();
}

/* SessionCache */

std::shared_ptr<SessionCache>
SessionCache::create (const CacheConfig &config,
    std::shared_ptr<DeviceLocks> device_locks, DeviceMemoryProbe probe)
{
  if (!device_locks)
    throw std::invalid_argument ("a session cache needs device locks");
  return std::shared_ptr<SessionCache> (
      new SessionCache (config, std::move (device_locks), std::move (probe)));
}

SessionCache::SessionCache (const CacheConfig &config,
    std::shared_ptr<DeviceLocks> device_locks, DeviceMemoryProbe probe)
    : config_ (config), device_locks_ (std::move (device_locks)), probe_ (std::move (probe))
{
}

SessionCache::~SessionCache () = default;

std::shared_ptr<Entry>
SessionCache::entry_for (const ReplicaSpec &spec)
{
  auto it = entries_.find (spec.key);
  std::shared_ptr<Entry> entry;
  if (it != entries_.end ()) {
    entry = it->second;
  } else {
    entry = std::make_shared<Entry> (spec);
    entries_.emplace (spec.key, entry);
  }
  while (entry->redirect)
    entry = entry->redirect;
  return entry;
}

void
SessionCache::add_consumer (const std::shared_ptr<Entry> &entry)
{
  std::lock_guard<std::mutex> lock (entry->mu);
  entry->consumers++;
}

EntryRef
SessionCache::acquire (const ReplicaSpec &spec, const ReplicaSpec *fallback)
{
  EntryRef ref;
  ref.cache_ = shared_from_this ();
  {
    std::lock_guard<std::mutex> lock (cache_mu_);
    ref.entry_ = entry_for (spec);
    add_consumer (ref.entry_);
  }

  try {
    ensure_replica (ref.entry_);
  } catch (const std::exception &e) {
    if (!fallback || !(ref.entry_->spec.key == spec.key)) {
      ref.reset ();
      throw;
    }
    g_warning ("onnxruntime session cache: creating %s failed (%s), falling back to %s",
        spec.key.canonical.c_str (), e.what (), fallback->key.canonical.c_str ());
    ref.poison (*fallback);
    try {
      ensure_replica (ref.entry_);
    } catch (...) {
      ref.reset ();
      throw;
    }
  }
  evict ();
  return ref;
}

void
SessionCache::ensure_replica (const std::shared_ptr<Entry> &entry)
{
  std::unique_lock<std::mutex> lock (entry->mu);
  for (;;) {
    if (!entry->replicas.empty ()) {
      hits_++;
      return;
    }
    if (entry->building == 0)
      break;
    entry->cv.wait (lock);
  }
  misses_++;
  entry->building++;
  lock.unlock ();

  std::unique_ptr<ReplicaSlot> slot;
  try {
    slot = build (entry->spec);
  } catch (...) {
    lock.lock ();
    entry->building--;
    entry->cv.notify_all ();
    throw;
  }

  lock.lock ();
  entry->building--;
  entry->replicas.push_back (std::move (slot));
  entry->cv.notify_all ();
}

std::unique_ptr<ReplicaSlot>
SessionCache::build (const ReplicaSpec &spec)
{
  auto slot = std::make_unique<ReplicaSlot> ();
  bool measured = false;
  slot->device = spec.device;

  if (spec.device >= 0) {
    std::lock_guard<std::mutex> lock (device_locks_->of (spec.device));
    measured = measure_growth (probe_, spec.device,
        [&slot, &spec] () { slot->replica = spec.create (); }, slot->bytes);
  } else {
    slot->replica = spec.create ();
  }
  if (!slot->replica)
    throw std::runtime_error ("session factory returned no session for " + spec.key.canonical);
  if (!measured && spec.host_bytes)
    slot->bytes = spec.host_bytes ();

  slot->last_used = tick ();
  sessions_created_++;
  g_info ("onnxruntime session cache: created session for %s (%" G_GUINT64_FORMAT " bytes)",
      spec.key.canonical.c_str (), (guint64) slot->bytes);
  return slot;
}

void
SessionCache::grow_async (const std::shared_ptr<Entry> &entry)
{
  grows_++;
  grows_in_flight_++;
  auto finish = [] (const std::shared_ptr<SessionCache> &self,
                    const std::shared_ptr<Entry> &entry, std::unique_ptr<ReplicaSlot> slot) {
    std::unique_ptr<ReplicaSlot> discard;
    {
      std::lock_guard<std::mutex> lock (entry->mu);
      entry->building--;
      if (!slot) {
        entry->grow_failed = true;
        self->grow_failures_++;
      } else if (entry->poisoned) {
        discard = std::move (slot);
      } else {
        entry->replicas.push_back (std::move (slot));
      }
      entry->cv.notify_all ();
    }
    if (discard) {
      std::vector<std::unique_ptr<ReplicaSlot>> slots;
      slots.push_back (std::move (discard));
      self->destroy_slots (slots);
      self->sessions_destroyed_++;
    }
    self->evict ();
    self->grows_in_flight_--;
  };

  std::shared_ptr<SessionCache> self = shared_from_this ();
  try {
    std::thread ([self, entry, finish] () {
      std::unique_ptr<ReplicaSlot> slot;
      try {
        slot = self->build (entry->spec);
      } catch (const std::exception &e) {
        g_warning ("onnxruntime session cache: growing %s failed: %s",
            entry->spec.key.canonical.c_str (), e.what ());
      }
      finish (self, entry, std::move (slot));
    }).detach ();
  } catch (const std::system_error &e) {
    g_warning ("onnxruntime session cache: cannot start a thread to grow %s: %s",
        entry->spec.key.canonical.c_str (), e.what ());
    finish (self, entry, nullptr);
  }
}

Lease
SessionCache::lease (const std::shared_ptr<Entry> &entry, std::shared_ptr<Entry> &redirect)
{
  const unsigned max_replicas = std::max (1u, config_.max_replicas_per_key);
  const bool exclusive = entry->spec.concurrency == Concurrency::Exclusive;
  bool grow_requested = false;
  std::unique_lock<std::mutex> lock (entry->mu);

  /* exclusive replicas are handed out first come, first served: a waiter holds a ticket,
   * and a consumer that returns a replica while others wait goes to the back */
  bool queued = false;
  uint64_t ticket = 0;
  auto dequeue = [&] () {
    if (queued) {
      entry->waiting.erase (std::find (entry->waiting.begin (), entry->waiting.end (), ticket));
      queued = false;
      entry->cv.notify_all ();
    }
  };

  for (;;) {
    if (entry->redirect) {
      dequeue ();
      redirect = entry->redirect;
      return Lease ();
    }

    const bool my_turn = !exclusive
                         || (queued ? entry->waiting.front () == ticket : entry->waiting.empty ());
    ReplicaSlot *pick = nullptr;
    if (!my_turn) {
    } else if (!exclusive) {
      for (auto &slot : entry->replicas)
        if (!pick || slot->leases < pick->leases)
          pick = slot.get ();
    } else {
      /* a free replica this thread already warmed up (no new graph capture), then the
       * most recently used one, so surplus replicas age out */
      const std::thread::id self = std::this_thread::get_id ();
      bool pick_warm = false;
      for (auto &slot : entry->replicas) {
        if (slot->leases > 0)
          continue;
        bool warm = slot->runs_of (self) != nullptr;
        if (!pick || (warm && !pick_warm) || (warm == pick_warm && slot->last_used > pick->last_used)) {
          pick = slot.get ();
          pick_warm = warm;
        }
      }
    }

    if (pick) {
      if (queued) {
        entry->waiting.pop_front ();
        queued = false;
        /* the next in line may take another free replica */
        entry->cv.notify_all ();
      }
      pick->leases++;
      pick->last_used = tick ();
      Lease lease;
      lease.cache_ = shared_from_this ();
      lease.entry_ = entry;
      lease.slot_ = pick;
      lease.first_use_ = !pick->used;
      lease.contended_ = entry->consumers > 1;
      pick->used = true;
      if (entry->spec.serialized_warmup_runs > 0) {
        unsigned *runs = pick->runs_of (std::this_thread::get_id ());
        if (!runs) {
          pick->runs_by_thread.emplace_back (std::this_thread::get_id (), 0);
          runs = &pick->runs_by_thread.back ().second;
        }
        lease.serialize_ = (*runs)++ < entry->spec.serialized_warmup_runs;
      }
      return lease;
    }

    if (exclusive && !queued) {
      ticket = entry->next_ticket++;
      entry->waiting.push_back (ticket);
      queued = true;
      continue;
    }

    if (entry->replicas.empty () && entry->building == 0) {
      lock.unlock ();
      try {
        ensure_replica (entry);
      } catch (...) {
        lock.lock ();
        dequeue ();
        throw;
      }
      lock.lock ();
      continue;
    }

    if (!grow_requested && exclusive
        && !entry->grow_failed && entry->replicas.size () + entry->building < max_replicas) {
      grow_requested = true;
      entry->building++;
      lock.unlock ();
      grow_async (entry);
      lock.lock ();
      continue;
    }

    entry->cv.wait (lock);
  }
}

void
SessionCache::return_lease (Lease &lease)
{
  const std::shared_ptr<Entry> &entry = lease.entry_;
  std::unique_ptr<ReplicaSlot> discard;
  {
    std::lock_guard<std::mutex> lock (entry->mu);
    lease.slot_->leases--;
    lease.slot_->last_used = tick ();
    if (entry->poisoned && lease.slot_->leases == 0)
      discard = entry->take (lease.slot_);
    entry->cv.notify_all ();
  }
  if (discard) {
    std::vector<std::unique_ptr<ReplicaSlot>> slots;
    slots.push_back (std::move (discard));
    destroy_slots (slots);
    sessions_destroyed_++;
  }
}

void
SessionCache::run_measured (Lease &lease, const std::function<void ()> &fn)
{
  const ReplicaSpec &spec = lease.entry_->spec;
  const bool measure = lease.first_use_ && probe_;
  if (spec.device < 0 || !(measure || lease.serialize_)) {
    fn ();
    return;
  }
  lease.first_use_ = false;
  lease.serialize_ = false;

  uint64_t grown = 0;
  {
    std::lock_guard<std::mutex> lock (device_locks_->of (spec.device));
    measure_growth (measure ? probe_ : DeviceMemoryProbe (), spec.device, fn, grown);
  }
  if (grown == 0)
    return;
  {
    std::lock_guard<std::mutex> lock (lease.entry_->mu);
    lease.slot_->bytes += grown;
  }
  evict ();
}

bool
SessionCache::inspect (const std::shared_ptr<Entry> &entry, const std::function<void (const Replica &)> &fn)
{
  std::unique_lock<std::mutex> lock (entry->mu);
  for (;;) {
    if (entry->redirect)
      return false;
    if (!entry->replicas.empty ())
      break;
    lock.unlock ();
    ensure_replica (entry);
    lock.lock ();
  }
  fn (*entry->replicas.front ()->replica);
  return true;
}

void
SessionCache::release_consumer (const std::shared_ptr<Entry> &entry)
{
  {
    std::lock_guard<std::mutex> cache_lock (cache_mu_);
    std::lock_guard<std::mutex> lock (entry->mu);
    entry->consumers--;
  }
  evict ();
}

std::shared_ptr<Entry>
SessionCache::follow (const std::shared_ptr<Entry> &from)
{
  std::shared_ptr<Entry> to = from;
  {
    std::lock_guard<std::mutex> cache_lock (cache_mu_);
    while (to->redirect)
      to = to->redirect;
    {
      std::lock_guard<std::mutex> lock (from->mu);
      from->consumers--;
    }
    add_consumer (to);
  }
  evict ();
  return to;
}

std::shared_ptr<Entry>
SessionCache::poison (const std::shared_ptr<Entry> &entry, const ReplicaSpec &fallback)
{
  if (entry->spec.key == fallback.key)
    throw std::logic_error ("cannot fall back to the same key " + fallback.key.canonical);

  std::shared_ptr<Entry> to;
  std::vector<std::unique_ptr<ReplicaSlot>> discard;
  {
    std::lock_guard<std::mutex> cache_lock (cache_mu_);
    if (entry->redirect) {
      to = entry->redirect;
      while (to->redirect)
        to = to->redirect;
    } else {
      to = entry_for (fallback);
      std::lock_guard<std::mutex> lock (entry->mu);
      entry->poisoned = true;
      entry->redirect = to;
      entry->take_idle (discard);
      entry->cv.notify_all ();
      poisons_++;
      g_warning ("onnxruntime session cache: %s is broken, redirecting its consumers to %s",
          entry->spec.key.canonical.c_str (), to->spec.key.canonical.c_str ());
    }
    {
      std::lock_guard<std::mutex> lock (entry->mu);
      entry->consumers--;
    }
    add_consumer (to);
  }
  sessions_destroyed_ += discard.size ();
  destroy_slots (discard);
  evict ();
  return to;
}

void
SessionCache::evict ()
{
  std::vector<std::unique_ptr<ReplicaSlot>> victims;
  size_t evicted = 0;
  bool overcommitted = false;
  {
    std::lock_guard<std::mutex> cache_lock (cache_mu_);

    /* broken entries and, with Asap, unreferenced entries keep nothing idle */
    for (auto &kv : entries_) {
      Entry &entry = *kv.second;
      std::lock_guard<std::mutex> lock (entry.mu);
      if (entry.poisoned || (entry.consumers == 0 && config_.strategy == CacheStrategy::Asap))
        entry.take_idle (victims);
    }
    evicted = victims.size ();

    uint64_t count = 0, bytes = 0;
    for (auto &kv : entries_) {
      Entry &entry = *kv.second;
      std::lock_guard<std::mutex> lock (entry.mu);
      count += entry.replicas.size ();
      for (auto &slot : entry.replicas)
        bytes += slot->bytes;
    }
    auto over = [&] () {
      return (config_.max_sessions && count > config_.max_sessions)
             || (config_.max_bytes && bytes > config_.max_bytes);
    };

    while (over ()) {
      /* idle replicas of unreferenced entries first, then surplus idle replicas, oldest first */
      Entry *victim_entry = nullptr;
      ReplicaSlot *victim = nullptr;
      int victim_tier = 2;
      for (auto &kv : entries_) {
        Entry &entry = *kv.second;
        std::lock_guard<std::mutex> lock (entry.mu);
        int tier = entry.consumers == 0 ? 0 : (entry.replicas.size () > 1 ? 1 : 2);
        if (tier > victim_tier || tier == 2)
          continue;
        for (auto &slot : entry.replicas) {
          if (slot->leases > 0)
            continue;
          if (tier < victim_tier || slot->last_used < victim->last_used) {
            victim_entry = &entry;
            victim = slot.get ();
            victim_tier = tier;
          }
        }
      }
      if (!victim) {
        overcommitted = true;
        break;
      }
      /* the scan let go of the entry: it may have been leased or removed since */
      std::lock_guard<std::mutex> lock (victim_entry->mu);
      if (victim->leases > 0)
        continue;
      uint64_t victim_bytes = victim->bytes;
      std::unique_ptr<ReplicaSlot> taken = victim_entry->take (victim);
      if (!taken)
        continue;
      count--;
      bytes -= victim_bytes;
      victims.push_back (std::move (taken));
      evicted++;
    }

    /* a poisoned entry keeps redirecting to its target, so the target stays */
    std::unordered_set<const Entry *> redirect_targets;
    for (auto &kv : entries_)
      if (kv.second->redirect)
        redirect_targets.insert (kv.second->redirect.get ());

    for (auto it = entries_.begin (); it != entries_.end ();) {
      Entry &entry = *it->second;
      bool unused;
      {
        std::lock_guard<std::mutex> lock (entry.mu);
        unused = !entry.poisoned && entry.consumers == 0 && entry.building == 0
                 && entry.replicas.empty () && !redirect_targets.count (&entry);
      }
      /* erasing may destroy the Entry, so never while holding its mutex */
      if (unused)
        it = entries_.erase (it);
      else
        ++it;
    }
  }

  if (overcommitted) {
    overcommits_++;
    g_info ("onnxruntime session cache: over its limits (max-sessions=%" G_GUINT64_FORMAT
            ", max-bytes=%" G_GUINT64_FORMAT ") with every replica in use",
        (guint64) config_.max_sessions, (guint64) config_.max_bytes);
  }
  evictions_ += evicted;
  sessions_destroyed_ += victims.size ();
  destroy_slots (victims);
}

CacheStats
SessionCache::stats () const
{
  CacheStats stats;
  {
    std::lock_guard<std::mutex> cache_lock (cache_mu_);
    stats.entries = entries_.size ();
    for (auto &kv : entries_) {
      Entry &entry = *kv.second;
      std::lock_guard<std::mutex> lock (entry.mu);
      stats.replicas += entry.replicas.size ();
      stats.consumers += entry.consumers;
      for (auto &slot : entry.replicas)
        stats.bytes += slot->bytes;
    }
  }
  stats.sessions_created = sessions_created_;
  stats.sessions_destroyed = sessions_destroyed_;
  stats.hits = hits_;
  stats.misses = misses_;
  stats.evictions = evictions_;
  stats.overcommits = overcommits_;
  stats.grows = grows_;
  stats.grow_failures = grow_failures_;
  stats.poisons = poisons_;
  stats.grows_in_flight = grows_in_flight_;
  return stats;
}

} /* namespace tensor_filter_onnxruntime */
} /* namespace nnstreamer */
