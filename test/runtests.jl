using JuliaProbo
using Plots
using Test
using JuliaFormatter

ENV["GKSwstype"] = "nul"

format("../")

include("ch03_test.jl")
include("ch04_test.jl")
include("ch05_test.jl")
include("ch06_test.jl")
include("ch07_test.jl")
