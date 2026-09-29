/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    nnstreamer_shared_slot.h
 * @brief   A ref-counted slot an application hands to sub-plugins, which fill it lazily.
 *
 * An application that wants sub-plugin state shared across tensor_filter
 * instances (e.g. the onnxruntime session cache) creates an empty slot and
 * places it in a GstContext. The first sub-plugin instance that finds the slot
 * fills it; every later instance gets the same data. The data is destroyed
 * with the last reference to the slot.
 */
#ifndef __NNSTREAMER_SHARED_SLOT_H__
#define __NNSTREAMER_SHARED_SLOT_H__

#include <glib-object.h>
#include "nnstreamer_api.h"

G_BEGIN_DECLS

typedef struct _NnsSharedSlot NnsSharedSlot;

/** @brief Creates the data of a slot; called at most once per slot. */
typedef gpointer (*NnsSharedSlotCreateFunc) (gpointer user_data);

extern NNS_API GType nns_shared_slot_get_type (void);
#define NNS_TYPE_SHARED_SLOT (nns_shared_slot_get_type ())

/** @brief An empty slot. */
extern NNS_API NnsSharedSlot *nns_shared_slot_new (void);

extern NNS_API NnsSharedSlot *nns_shared_slot_ref (NnsSharedSlot * slot);

/** @brief Drops a reference; the last one destroys the data with its destroy notify. */
extern NNS_API void nns_shared_slot_unref (NnsSharedSlot * slot);

/**
 * @brief The data of slot, created by create (with user_data) if the slot is still empty.
 * @return The data, or NULL if create returned NULL (the slot then stays empty).
 */
extern NNS_API gpointer nns_shared_slot_get_or_create (NnsSharedSlot * slot,
    NnsSharedSlotCreateFunc create, gpointer user_data, GDestroyNotify destroy);

/** @brief The data of slot, NULL while empty. */
extern NNS_API gpointer nns_shared_slot_peek (NnsSharedSlot * slot);

G_END_DECLS

#endif /* __NNSTREAMER_SHARED_SLOT_H__ */
