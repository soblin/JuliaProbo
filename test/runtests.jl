using JuliaProbo
using Plots
using Test
using JuliaFormatter

ENV["GKSwstype"] = "nul"
GUI = false
FORMATTER = true

if FORMATTER
    format("../src")
    format_file("../test/ch03_test.jl")
    format_file("../test/ch04_test.jl")
    format_file("../test/ch05_test.jl")
    format_file("../test/ch06_test.jl")
    format_file("../test/ch07_test.jl")
    format_file("../test/ch08_test.jl")
    format_file("../test/ch09_test.jl")
    # Somehow this file fails
    # format_file("../test/ch10_test.jl")
    format_file("../test/ch12_test.jl")
end

if false
    include("ch03_test.jl")
    include("ch04_test.jl")
    include("ch05_test.jl")
    include("ch06_test.jl")
    include("ch07_test.jl")
    include("ch08_test.jl")
    include("ch09_test.jl")
    include("ch10_test.jl")
    include("ch12_test.jl")
end
