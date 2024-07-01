module MultiLinearInterpolationsForwardDiffExt

import ForwardDiff
import MultiLinearInterpolations

function MultiLinearInterpolations._reinterpret_dual(
    DualType::Type{<:ForwardDiff.Dual},
    v::AbstractArray{T},
    n::Int,
) where {T}
    n_elems = div(sizeof(DualType), sizeof(T)) * n
    if n_elems > length(v)
        @warn "Resizing..."
        resize!(v, n_elems)
    end
    reinterpret(DualType, view(v, 1:n_elems))
end

end # module