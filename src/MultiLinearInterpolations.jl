module MultiLinearInterpolations

@fastmath _linear_interpolate(y1, y2, θ) = (1 - θ) * y1 + θ * y2
function _linear_interpolate!(
    out::AbstractArray{<:Number},
    y1::Union{<:Number,AbstractArray{<:Number}},
    y2::Union{<:Number,AbstractArray{<:Number}},
    θ,
)
    @. out = (1 - θ) * y1 + θ * y2
end
function _linear_interpolate!(out::AbstractArray{T}, y1::T, y2::T, θ) where {T}
    _linear_interpolate!(out[1], y1, y2, θ)
end
function _linear_interpolate!(
    out::AbstractArray{T},
    y1::AbstractArray{T},
    y2::AbstractArray{T},
    θ,
) where {T}
    for i in eachindex(y1)
        _linear_interpolate!(out[i], y1[i], y2[i], θ)
    end
end
@inline function _linear_interpolate(arr::AbstractVector, idx, θ)
    _linear_interpolate(arr[idx], arr[idx+1], θ)
end

restructure(::Number, vals::AbstractVector) = first(vals)
restructure(::AbstractArray, vals::AbstractVector) = vals
restructure(::NTuple{N}, vals::AbstractVector) where {N} = ((vals[i] for i = 1:N)...,)

function _tuple_set(tuple::NTuple{N}, index, v)::NTuple{N} where {N}
    if index == 1
        (v, tuple[2:end]...)
    elseif index == N
        (tuple[1:end-1]..., v)
    else
        (tuple[1:index-1]..., v, tuple[index+1:end]...)
    end
end

# will probably be resized anyway, so let's not fret about using 
# ForwardDiff's estimated chunk size
function _dual_size(len::Int; chunk_size = 8)
    cs = prod((chunk_size * ones(Int, 1)) .+ 1)
    cs * len
end

struct MultilinearInterpolator{D,T}
    indices::Vector{NTuple{D,Int}}
    weights::Vector{T}
    output::Vector{T}
    output_len::Int
    function MultilinearInterpolator{D}(
        values::AbstractArray{C};
        T = Float64,
        kwargs...,
    ) where {D,C}
        size_points = 2^D
        indices = Vector{NTuple{D,Int}}(undef, size_points)
        weights = zeros(T, _dual_size(D; kwargs...))

        len = if C <: Number
            1
        elseif eltype(C) <: Number
            length(values[1])
        else
            sum(fieldnames(C)) do f
                length(getproperty(values[1], f))
            end
        end

        output = Vector{T}(undef, _dual_size(len + 1; kwargs...))
        new{D,T}(indices, weights, output, len)
    end
end

_reinterpret_dual(::Type, v::AbstractArray, n::Int) = view(v, 1:n)

function update_indices!(
    cache::MultilinearInterpolator{D},
    axes::NTuple{D},
    x::NTuple{D,<:Number},
) where {D}
    its = ((1:D)...,)

    weights = _reinterpret_dual(typeof(first(x)), cache.weights, D)

    map(its) do I
        stride = 2^(D - I)
        ax = axes[I]
        x0 = x[I]

        i = min(lastindex(ax), searchsortedfirst(ax, x0))
        i1, i2 = if i == 1
            1, 2
        else
            i - 1, i
        end

        weights[I] = (x0 - ax[i1]) / (ax[i2] - ax[i1])

        for (q, j) in enumerate(range(1, lastindex(cache.indices), step = stride))
            for k = j:j+stride-1
                tup = cache.indices[k]
                if !iseven(q)
                    cache.indices[k] = _tuple_set(tup, I, i1)
                else
                    cache.indices[k] = _tuple_set(tup, I, i2)
                end
            end
        end

        nothing
    end
    cache.indices
end

function _build_multilinear_expression(D::Int, field_name)
    function _lerp(y1, y2, w)
        :($(y2) * $(w) + $(y1) * (1 - $(w)))
    end
    assignments = []
    _index(i) =
        if !isnothing(field_name)
            sym = Base.gensym()
            push!(
                assignments,
                :(
                    $(sym) = getproperty(
                        values[cache.indices[$(i)]...],
                        $(Meta.quot(field_name)),
                    )
                ),
            )
            sym
        else
            :(values[cache.indices[$(i)]...])
        end
    _weight(i) = :(weights[$(i)])

    weight_index = D

    # get the knots
    knots = map(1:2^(D-1)) do d
        i = d * 2
        _lerp(_index(i - 1), _index(i), _weight(weight_index))
    end

    while length(knots) > 1
        weight_index -= 1
        knots = map(range(1, lastindex(knots), step = 2)) do i
            _lerp(knots[i], knots[i+1], _weight(weight_index))
        end
    end

    assignments, first(knots)
end

@inline @generated function _interpolate_cache!(
    cache::MultilinearInterpolator{D},
    values::AbstractArray{T,D},
    p::NTuple,
) where {D,T}
    assignments = []
    exprs = if T <: Number || eltype(T) <: Number
        _, interp = _build_multilinear_expression(D, nothing)
        expr = quote
            @. output = $(interp)
        end
        [expr]
    else
        interps = (_build_multilinear_expression(D, i) for i in fieldnames(T))
        map(zip(interps, fieldnames(T))) do I
            (assign, interp), f = I
            append!(assignments, assign)
            sym = Base.gensym()
            quote
                start = stop + 1
                shape = size(getproperty(values[cache.indices[1]...], $(Meta.quot(f))))
                @views begin
                    $(sym) = if length(shape) > 0
                        stop = start + prod(shape) - 1
                        reshape(output[start:stop], shape)
                    else
                        stop = start
                        output[start:stop]
                    end
                    @. $(sym) = $(interp)
                end
            end
        end
    end
    quote
        begin
            weights = _reinterpret_dual(eltype(p), cache.weights, D)
            output = _reinterpret_dual(eltype(p), cache.output, cache.output_len)

            start::Int = 0
            stop::Int = 0
            $(assignments...)
            $(exprs...)

            output
        end
    end
end

function interpolate!(
    cache::MultilinearInterpolator{D},
    axes::NTuple{D},
    vals::AbstractArray{T,D},
    x::NTuple{D,<:Number},
) where {D,T}
    update_indices!(cache, axes, x)
    output = _interpolate_cache!(cache, vals, x)
    restructure(first(vals), output)
end

export MultilinearInterpolator, interpolate!


end # module MultiLinearInterpolations
