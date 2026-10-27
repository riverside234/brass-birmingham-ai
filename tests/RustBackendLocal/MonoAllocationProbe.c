/* Diagnostic only: Mono's public profiler callback, no runtime patching.
 * API: mono/metadata/profiler.h, profiler-events.h and object.h.
 * Counts requested managed object bytes, not retained heap or allocator slack.
 * Allocation instrumentation must already be enabled by the embedding host.
 * A managed self-test rejects a silent/late/incomplete callback installation.
 */
#include <windows.h>
#include <stdint.h>
typedef void *(*create_fn)(void *);
typedef void (*set_fn)(void *, void (*)(void *, void *));
typedef unsigned int (*size_fn)(void *);
static size_fn object_size;
static void *handle;
static _Thread_local uint64_t bytes, objects;
static void allocation(void *prof, void *object) {
    (void)prof;
    bytes += object_size(object);
    objects++;
}
__declspec(dllexport) int brass_alloc_start(void) {
    if (handle) return 1;
    HMODULE mono=GetModuleHandleW(L"mono-2.0-bdwgc.dll");
    if (!mono) return 0;
    create_fn create=(create_fn)GetProcAddress(mono,"mono_profiler_create");
    set_fn set=(set_fn)GetProcAddress(mono,"mono_profiler_set_gc_allocation_callback");
    object_size=(size_fn)GetProcAddress(mono,"mono_object_get_size");
    if (!create || !set || !object_size) return 0;
    handle=create(NULL);
    if (!handle) return 0;
    set(handle,allocation);
    return 1;
}
__declspec(dllexport) uint64_t brass_alloc_bytes(void) { return bytes; }
__declspec(dllexport) uint64_t brass_alloc_objects(void) { return objects; }
