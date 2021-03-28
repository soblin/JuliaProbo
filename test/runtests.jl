using JuliaProbo
using Plots
using Test

ENV["GKSwstype"] = "nul"
GUI = false

include("ch03_test.jl")
include("ch04_test.jl")
include("ch05_test.jl")
include("ch06_test.jl")
include("ch07_test.jl")
include("ch08_test.jl")
include("ch09_test.jl")
include("ch10_test.jl")
include("ch12_test.jl")
