abstract type AbstractObject end
abstract type AbstractLandmark <: AbstractObject end
abstract type AbstractSensor end
abstract type AbstractCamera <: AbstractSensor end
abstract type AbstractAgent end
abstract type AbstractEstimator end
abstract type AbstractParticle end
abstract type AbstractMcl <: AbstractEstimator end
abstract type AbstractResetMcl <: AbstractMcl end
