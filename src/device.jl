abstract type AbstractDevice end
struct CPU <: AbstractDevice end
struct GPU <: AbstractDevice end

"""
    to_device(device::AbstractDevice, x)

Transfer data to the specified device.
"""
to_device(::CPU, x) = x |> cpu
to_device(::GPU, x) = x |> gpu

"""
    array_type(device::AbstractDevice, T::Type)

Get the appropriate array type for the device.
"""
array_type(::CPU, T::Type) = Array{T}
array_type(::GPU, T::Type) = CuArray{T}

"""
    create_array(device::AbstractDevice, T::Type, dims...)

Create an array on the specified device.
"""
function create_array(device::AbstractDevice, T::Type, dims...)
    return array_type(device, T)(undef, dims...)
end

"""
    randn_device(device::AbstractDevice, T::Type, dims...)

Create a random normal array on the specified device.
"""
function randn_device(device::CPU, T::Type, dims...)
    return randn(T, dims...)
end

function randn_device(device::GPU, T::Type, dims...)
    return CUDA.randn(T, dims...)
end

"""
    device_string(device::AbstractDevice)

Get a string representation of the device.
"""
device_string(::CPU) = "cpu"
device_string(::GPU) = "cuda"

function pick_gpu_by_nvidiasmi(min_free_mb::Int=4096)
    raw = read(`nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits`, String)
    free = [(parse(Int, split(line, ",")[2]), parse(Int, split(line, ",")[1])) for line in split(raw, '\n') if !isempty(line)]
    if isempty(free); return nothing; end
    best = argmax(first, free)
    best_free, best_idx = best
    return best_free >= min_free_mb ? best_idx : nothing
end

"""
    select_device(device_str::String)

Get device from string specification.

# Arguments
- `device_str::String`: Device specification ("cpu" or "cuda"/"gpu")

# Returns
- Device object (CPU() or GPU())
"""
function select_device(device_str::String)
    device_lower = lowercase(device_str)
    if device_lower == "cpu"
        return CPU()
    elseif device_lower in ["cuda", "gpu"]
        if !CUDA.functional()
            @warn "CUDA is not available, falling back to CPU"
            return CPU()
        end
        device_idx = pick_gpu_by_nvidiasmi()
        CUDA.device!(device_idx)
        @info "Selected GPU: $device_idx"
        return GPU()
    else
        throw(ArgumentError("Unknown device: $device_str. Use 'cpu' or 'cuda'/'gpu'"))
    end
end

"""
    cpu(x)

Move data to CPU.
"""
cpu(x::Array) = x
cpu(x::CuArray) = Array(x)
cpu(x) = x

"""
    gpu(x)

Move data to GPU.
"""
gpu(x::Array) = CuArray(x)
gpu(x::CuArray) = x
gpu(x) = x

