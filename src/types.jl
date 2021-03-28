abstract type AbstractObject end
abstract type AbstractWorld end
abstract type AbstractLandmark <: AbstractObject end
abstract type AbstractSensor end
abstract type AbstractCamera <: AbstractSensor end
abstract type AbstractRobot <: AbstractObject end
abstract type AbstractAgent end
abstract type AbstractEstimator end
abstract type AbstractParticle end
abstract type AbstractMcl <: AbstractEstimator end
abstract type AbstractResetMcl <: AbstractMcl end
abstract type AbstractMDPAgent <: AbstractAgent end
# KalmanFilter isa AbstractEstimator
# Mcl, KldMcl, FastSlam1, FastSlam2 isa AbstractMcl
# ResetMcl, AMcl isa AbstractResetMcl
