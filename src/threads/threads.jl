module ThreadsImpl

import Base: Callable
using JACC

struct ThreadsBackend end

include("array.jl")
include("multi.jl")
include("async.jl")
include("experimental/experimental.jl")

JACC.get_backend(::Val{:threads}) = ThreadsBackend()

function _maybe_threaded(ex)
    quote
        if Threads.nthreads() == 1
            $ex
        else
            Threads.@threads :static $ex
        end
    end
end

macro maybe_threaded(ex)
    esc(_maybe_threaded(ex))
end

synchronize(::ThreadsBackend) = nothing

JACC.default_stream(::Type{ThreadsBackend}) = nothing

function JACC.parallel_for(::ThreadsBackend, N::Integer, f::Callable, x...)
    @maybe_threaded for i in 1:N
        f(i, x...)
    end
end

function JACC.parallel_for(
        ::LaunchSpec{ThreadsBackend}, N::Integer, f::Callable, x...)
    parallel_for(ThreadsBackend(), N, f, x...)
end

function JACC.parallel_for(
        ::ThreadsBackend, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    @maybe_threaded for j in 1:N
        for i in 1:M
            f(i, j, x...)
        end
    end
end

function JACC.parallel_for(
        ::LaunchSpec{ThreadsBackend}, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    parallel_for(ThreadsBackend(), (M, N), f, x...)
end

function JACC.parallel_for(
        ::ThreadsBackend, (L, M, N)::NTuple{3, Integer}, f::Callable, x...)
    # only threaded at the first level (no collapse equivalent)
    @maybe_threaded for k in 1:N
        for j in 1:M
            for i in 1:L
                f(i, j, k, x...)
            end
        end
    end
end

function JACC.parallel_for(
        ::LaunchSpec{ThreadsBackend}, (L, M, N)::NTuple{3, Integer}, f::Callable, x...)
    parallel_for(ThreadsBackend(), (L, M, N), f, x...)
end

function JACC.parallel_reduce(
        ::ThreadsBackend, N::Integer, op, f::Callable, x...; init)
    ret = init
    tmp = fill(init, Threads.maxthreadid())
    @maybe_threaded for i in 1:N
        tmp[Threads.threadid()] = op.(tmp[Threads.threadid()], f(i, x...))
    end
    for i in 1:Threads.maxthreadid()
        ret = op.(ret, tmp[i])
    end
    return ret
end

function JACC.parallel_reduce(
        ::ThreadsBackend, (M, N)::Tuple{Integer, Integer}, op, f::Callable, x...; init)
    ret = init
    tmp = fill(init, Threads.maxthreadid())
    @maybe_threaded for j in 1:N
        for i in 1:M
            tmp[Threads.threadid()] = op.(
                tmp[Threads.threadid()], f(i, j, x...))
        end
    end
    for i in 1:Threads.maxthreadid()
        ret = op.(ret, tmp[i])
    end
    return ret
end

JACC.array_type(::ThreadsBackend) = Base.Array

JACC.array(::ThreadsBackend, x::Base.Array) = x

end
