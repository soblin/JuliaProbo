using JuliaProbo
using Plots
using Test
using JuliaFormatter

ENV["GKSwstype"] = "nul"
GUI = false

format("../src")
format("../test")

#=include("ch03_test.jl")
include("ch04_test.jl")
include("ch05_test.jl")
include("ch06_test.jl")
include("ch07_test.jl")
include("ch08_test.jl")
include("ch09_test.jl")=#
include("ch10_test.jl")
