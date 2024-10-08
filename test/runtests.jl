using Test
using ForwardDiff, StaticArrays
using MultiLinearInterpolations

v = zeros(Float64, (3, 2, 2))

@inferred MultilinearInterpolator{2}(v)

X1 = [0.0, 1.0]
X2 = [1.0, 2.0]
vals = reshape(
    NTuple{3,Float64}[
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, -1.0),
        (2.0, 2.0, 2.0),
        (-2.0, -2.0, -2.0),
    ],
    (2, 2),
)

cache = MultilinearInterpolator{2}(vals)
interpolate!(cache, (X1, X2), vals, (0.0, 1.5))

@test interpolate!(cache, (X1, X2), vals, (0.0, 1.0)) == (1.0, 1.0, 1.0)
@test interpolate!(cache, (X1, X2), vals, (1.0, 1.5)) == (-1.5, -1.5, -1.5)
@test interpolate!(cache, (X1, X2), vals, (0.5, 1.5)) == (0.0, 0.0, 0.0)

@inferred interpolate!(cache, (X1, X2), vals, (0.0, 1.5))

@allocated interpolate!(cache, (X1, X2), vals, (0.0, 1.5))

ff(x, y) = 3x^2 + x * y - sin(y)
ff(x) = SVector(ff(x[1], x[2]))
X1 = collect(range(0, 1, 1000))
X2 = collect(range(0, 1, 1000))
vals = reshape([ff(x, y) for x in X1, y in X2], (length(X1), length(X2)))
cache = MultilinearInterpolator{2}(vals)

# check that dual cache works too
function _interpolate_wrapper(cache, X1, X2, vals)
    function _f(x)
        x1, x2 = x
        SVector(interpolate!(cache, (X1, X2), vals, (x1, x2)))
    end
end
interp_f = _interpolate_wrapper(cache, X1, X2, vals)
x0 = SVector(X1[3], X2[5])
@test ForwardDiff.jacobian(interp_f, x0) ≈ ForwardDiff.jacobian(ff, x0) atol =
    1e-2

# now single dimension edge case
X1 = [0.0, 1.0]
vals = [-1.0, 0.0]
cache = MultilinearInterpolator{1}(vals)

@test interpolate!(cache, (X1,), vals, (0.0,)) == -1.0
@test interpolate!(cache, (X1,), vals, (0.6,)) == -0.4

# check that dual cache works too
function _interpolate_wrapper(cache, X1, vals)
    function _f(x)
        interpolate!(cache, (X1,), vals, (x,))
    end
end
interp_f = _interpolate_wrapper(cache, X1, vals)
@test ForwardDiff.derivative(interp_f, 0.6) == 1
@inferred ForwardDiff.derivative(interp_f, 0.6)

# let's try it with arbitrary data structures
struct Thing{V<:AbstractVector,M<:AbstractMatrix}
    a::V
    b::M
end
Thing(a::Number, b::Number) = Thing([a], [b 0; 0 b])

function MultiLinearInterpolations.restructure(data::Thing, vals::AbstractVector)
    @views Thing(vals[1:length(data.a)], reshape(vals[length(data.a)+1:end], size(data.b)))
end

X1 = [0.0, 1.0]
X2 = [1.0, 2.0]
vals = reshape(
    [Thing(1.0, 1.0), Thing(-1.0, -1.0), Thing(2.0, 2.0), Thing(-2.0, -2.0)],
    (2, 2),
)

cache = MultilinearInterpolator{2}(vals)
intp = interpolate!(cache, (X1, X2), vals, (0.0, 1.5))
# do it twice to make sure no side-effects
intp = interpolate!(cache, (X1, X2), vals, (0.0, 1.5))
@test intp.a == [1.5]
@test intp.b == [1.5 0; 0 1.5]

# try higher dimensional interpolation

X1 = range(0.0, 1.0, 3)
X2 = range(1.0, 2.0, 4)
X3 = range(-1.0, 0.0, 2)

f3(x, y, z) = 2x + 3y + 7z
vals = [f3(x, y, z) for x in X1, y in X2, z in X3]

cache = MultilinearInterpolator{3}(vals)
intp = interpolate!(cache, (X1, X2, X3), vals, (0.5, 1.5, -0.5))
@test intp == f3(0.5, 1.5, -0.5)

intp = interpolate!(cache, (X1, X2, X3), vals, (1.0, 1.1, -0.1))
@test intp == f3(1.0, 1.1, -0.1)

intp = interpolate!(cache, (X1, X2, X3), vals, (1.0, 1.9, -0.1))
@test intp == f3(1.0, 1.9, -0.1)


using Aqua
# little bit of aqua
Aqua.test_undefined_exports(MultiLinearInterpolations)
Aqua.test_unbound_args(MultiLinearInterpolations)
Aqua.test_stale_deps(MultiLinearInterpolations)