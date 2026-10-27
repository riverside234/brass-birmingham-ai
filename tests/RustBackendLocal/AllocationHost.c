/* Diagnostic embedding host: enables the PUBLIC Mono profiler API before
 * managed code starts. Uses the exact shipped Mono DLL; no binary patches. */
#include <windows.h>
#include <stdio.h>
#include <stdint.h>
static _Thread_local uint64_t allocated, objects;
static unsigned int (*object_size)(void *);
static void on_alloc(void *prof,void *obj){(void)prof;allocated+=object_size(obj);objects++;}
__declspec(dllexport) uint64_t brass_bytes(void){return allocated;}
__declspec(dllexport) uint64_t brass_objects(void){return objects;}
#define API(type,name) type name=(type)GetProcAddress(mono,#name); if(!name){fprintf(stderr,"Missing %s\n",#name);return 3;}
typedef void *(*init_t)(const char *,const char *);
typedef void *(*open_t)(void *,const char *);
typedef int (*exec_t)(void *,void *,int,char **);
typedef void (*dirs_t)(const char *,const char *);
typedef void (*parse_t)(const char *);
typedef void *(*create_t)(void *);
typedef void (*callback_t)(void *,void (*)(void *,void *));
typedef int (*enable_t)(void);
int main(int argc,char **argv){
 if(argc<7){fprintf(stderr,"mono-dll managed-dir mono-etc probe.exe fixtures output\n");return 2;}
 HMODULE mono=LoadLibraryA(argv[1]);if(!mono){fprintf(stderr,"Mono load %lu\n",GetLastError());return 3;}
 API(dirs_t,mono_set_dirs);API(parse_t,mono_set_assemblies_path);API(parse_t,mono_config_parse);
 API(create_t,mono_profiler_create);API(callback_t,mono_profiler_set_gc_allocation_callback);API(enable_t,mono_profiler_enable_allocations);
 API(init_t,mono_jit_init_version);API(open_t,mono_domain_assembly_open);API(exec_t,mono_jit_exec);
 object_size=(unsigned int(*)(void*))GetProcAddress(mono,"mono_object_get_size");if(!object_size)return 3;
 void *prof=mono_profiler_create(NULL);mono_profiler_set_gc_allocation_callback(prof,on_alloc);
 if(!mono_profiler_enable_allocations()){fprintf(stderr,"Allocation enable refused\n");return 4;}
 mono_set_dirs(argv[2],argv[3]);mono_set_assemblies_path(argv[2]);mono_config_parse(NULL);
 void *domain=mono_jit_init_version("Brass allocation diagnostic","v4.0.30319");if(!domain)return 5;
 void *assembly=mono_domain_assembly_open(domain,argv[4]);if(!assembly)return 6;
 return mono_jit_exec(domain,assembly,argc-4,argv+4);
}
