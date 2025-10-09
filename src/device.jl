# Device management utilities for CPU/GPU support

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

