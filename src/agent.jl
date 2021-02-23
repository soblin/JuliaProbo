abstract type Object end

mutable struct Agent <: Object
    v_::Float64
    ω_::Float64
    function Agent(v::Float64, ω::Float64)
        new(v, ω)
    end
end

function decision(agent::Agent, observation::Any)
    return agent.v_, agent.ω_
end
