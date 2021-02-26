module JuliaProbo

import Plots: Plot, plot, plot!, scatter!, annotate!, quiver!, text
import Random: rand
import Distributions: Exponential, Normal, Uniform

export AbstractObject, AbstractSensor, AbstractAgent
export draw
export Agent, decision
export IdealRobot, RealRobot, state_transition
export Map
export IdealCamera, RealCamera, observations
export Landmark, World, push!

include("types.jl")
include("agent.jl")
include("robot.jl")
include("world.jl")
include("sensor.jl")

end # module
