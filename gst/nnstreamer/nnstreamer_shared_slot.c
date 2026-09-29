/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    nnstreamer_shared_slot.c
 * @brief   A ref-counted slot an application hands to sub-plugins, which fill it lazily.
 */

#include "nnstreamer_shared_slot.h"

struct _NnsSharedSlot
{
  GMutex lock;
  gpointer data;
  GDestroyNotify destroy;
};

static void
nns_shared_slot_clear (NnsSharedSlot * slot)
{
  if (slot->data && slot->destroy)
    slot->destroy (slot->data);
  g_mutex_clear (&slot->lock);
}

GType
nns_shared_slot_get_type (void)
{
  static gsize type = 0;
  if (g_once_init_enter (&type)) {
    /* an application may carry its own copy of this type; reuse whichever registered first */
    GType registered = g_type_from_name ("NnsSharedSlot");
    if (!registered) {
      registered = g_boxed_type_register_static ("NnsSharedSlot",
          (GBoxedCopyFunc) nns_shared_slot_ref,
          (GBoxedFreeFunc) nns_shared_slot_unref);
    }
    g_once_init_leave (&type, registered);
  }
  return type;
}

NnsSharedSlot *
nns_shared_slot_new (void)
{
  NnsSharedSlot *slot = g_atomic_rc_box_new0 (NnsSharedSlot);
  g_mutex_init (&slot->lock);
  return slot;
}

NnsSharedSlot *
nns_shared_slot_ref (NnsSharedSlot * slot)
{
  g_return_val_if_fail (slot != NULL, NULL);
  return g_atomic_rc_box_acquire (slot);
}

void
nns_shared_slot_unref (NnsSharedSlot * slot)
{
  if (slot)
    g_atomic_rc_box_release_full (slot, (GDestroyNotify) nns_shared_slot_clear);
}

gpointer
nns_shared_slot_get_or_create (NnsSharedSlot * slot,
    NnsSharedSlotCreateFunc create, gpointer user_data, GDestroyNotify destroy)
{
  gpointer data;

  g_return_val_if_fail (slot != NULL, NULL);
  g_return_val_if_fail (create != NULL, NULL);

  g_mutex_lock (&slot->lock);
  if (!slot->data) {
    slot->data = create (user_data);
    slot->destroy = slot->data ? destroy : NULL;
  }
  data = slot->data;
  g_mutex_unlock (&slot->lock);
  return data;
}

gpointer
nns_shared_slot_peek (NnsSharedSlot * slot)
{
  gpointer data;

  g_return_val_if_fail (slot != NULL, NULL);

  g_mutex_lock (&slot->lock);
  data = slot->data;
  g_mutex_unlock (&slot->lock);
  return data;
}
