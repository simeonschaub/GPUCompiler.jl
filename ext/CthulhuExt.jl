module CthulhuExt

using GPUCompiler: GPUInterpreter
using Cthulhu: Cthulhu

# Cthulhu 3 requires custom `AbstractInterpreter`s to opt into its provider
# interface; hook the `GPUInterpreter` up to the default provider (which wraps
# it in a `CthulhuInterpreter`, forwarding the GPU method table, inference
# parameters and world age) so that interactive reflection like
# `code_typed(job; interactive=true)` works. On Cthulhu 2 the `interp` keyword
# handles custom interpreters directly, so there is nothing to define.
@static if isdefined(Cthulhu, :AbstractProvider)
    Cthulhu.AbstractProvider(interp::GPUInterpreter) = Cthulhu.DefaultProvider(interp)
end

end
