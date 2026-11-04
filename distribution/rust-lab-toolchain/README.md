# Pinned qualification compiler

`microsoft.net.compilers.toolset.4.12.0.nupkg` is the unchanged Microsoft NuGet
package already used by this project's C# build tools. SHA-256:
`fe24ef31a6ffcb7c49383d2fd362763dee291ad9b9d98cc0c19ef80203b99ebc`.

B02 restores its `tasks/net472` compiler and dependencies into the new evidence
directory, checks startup, compiles the authored bootstrap, and checks one action
and checkpoint restoration before compiling Rust. No download or developer cache
is required. Linux still needs the existing Mono runtime and admitted private
game runtime. B04 reuses the verified B02 bootstrap.

The package metadata declares MIT licensing and Roslyn source commit
`dfa7fc6bdea31a858a402168384192b633c811fa`. The corresponding [license](https://github.com/dotnet/roslyn/blob/dfa7fc6bdea31a858a402168384192b633c811fa/License.txt)
is included as `License.txt`. No proprietary game binaries are bundled here.
