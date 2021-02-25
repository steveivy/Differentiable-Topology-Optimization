# top2D cantilever beam
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using SparseArrays
using LinearAlgebra
using StaticArrays
using ChainRulesCore
using ForwardDiff
using Flux
using Flux: gradient
using Flux.Optimise: update!
using Zygote: forwarddiff
using Random
using StatsFuns:logistic
using Printf
using Images
using FileIO
using InteractiveUtils
using Profile
using Colors
using Plots: plot


# element number in each direction
ex = 96::Int64
ey = 48::Int64
n_train = 100::Int64
WEIGHT_FRAC  = 0.3::Float64
penalty = 3
save_every = 10::Int64
resultFile = string("CANT_absU1_$WEIGHT_FRAC","_$ex","_$ey","_$penalty")
filter1 = 2::Int64
randomInput = 0::Int64
initialViolation = 0.0

nx = ex + 1
ny = ey + 1

# load points
fPoint = Int64(2*ny*ex+(ey/2+1)*2)

# relax parameters
# loss function
# model sturcutre
# material property

E = 1.0::Float64
Emin = 10.0^(-16)
ν = 0.3::Float64

etot = ex*ey

# element type
Kₑ = [45/91 5/28 -55/182 -5/364 -45/182 -5/28 5/91 5/364;
  5/28 45/91 5/364 5/91 -5/28 -45/182 -5/364 -55/182;
  -55/182 5/364 45/91 -5/28 5/91 -5/364 -45/182 5/28;
  -5/364 5/91 -5/28 45/91 5/364 -55/182 5/28 -45/182;
  -45/182 -5/28 5/91 5/364 45/91 5/28 -55/182 -5/364;
  -5/28 -45/182 -5/364 -55/182 5/28 45/91 5/364 5/91;
  5/91 -5/364 -45/182 5/28 -55/182 5/364 45/91 -5/28;
  5/364 -55/182 5/28 -45/182 -5/364 5/91 -5/28 45/91];

nodenrs = reshape([i for i in 1:nx*ny], ny, nx)
edofVec = reshape(2*nodenrs[1:end-1, 1:end-1].+1, ex*ey,1)
edofMat = repeat(edofVec,1,8)+repeat([0 1 2*ey.+[2 3 0 1] -2 -1], ex*ey,1)
iK = Array{Int64}(reshape(kron(edofMat, ones(8,1))', (64*ex*ey)))
jK = Array{Int64}(reshape(kron(edofMat, ones(1,8))', (64*ex*ey)))

# Load and boundary conditions
F = zeros((2*ny*nx,1))
F[fPoint] = -1.0
fixeddofs = sort(union([i for i in 1:1:2*ny]))
alldofs = [i for i in 1:2*ny*nx]
freedofs = setdiff(alldofs, fixeddofs)

dofIdx = Vector(undef, length(alldofs))
for i in 1:length(alldofs)
  if !issubset(alldofs[i], fixeddofs)
    temp = findall(x->x .< alldofs[i], fixeddofs)
    dofIdx[i] = alldofs[i]-length(temp)
  end
end

idx1 = Vector()
iK1 = Vector()
jK1 = Vector()
@time for i in 1:length(iK)
    if !issubset(iK[i], fixeddofs)
        if !issubset(jK[i], fixeddofs)
          push!(idx1, i)
          push!(iK1, dofIdx[iK[i]])
          push!(jK1, dofIdx[jK[i]])
        end
    end
end
iK = iK1
jK = jK1
fPoint = dofIdx[fPoint]
F = F[freedofs]


mybackslash(A::SparseMatrixCSC,B::AbstractVector) = A\B

function ChainRulesCore.rrule(::typeof(mybackslash), A, B)
    #C = A\B
    F = factorize(A)
    C = F \ B
    function mybackslash_pullback(dC)
        #dB = A'\dC
        dB = F \ collect(dC)
        return (NO_FIELDS, @thunk(-dB * C'), dB)
    end
    return C, mybackslash_pullback
end

function ChainRulesCore.rrule(::typeof(mybackslash), A::SparseMatrixCSC, b::AbstractVector)
    #c = A\b
    F = factorize(A)
    c = F \ b
    function mybackslash_pullback(dc)
        #db = A'\collect(dc)
        db = F \ collect(dc)
        # ^ When called through Zygote, `dc` is a `FillArrays.Fill`, and
        # `A'\dc` throws an error for such `dc`. The `collect()` here is a
        # workaround for this bug.

        dA = @thunk begin
            m,n,pp,ii = A.m,A.n,A.colptr,A.rowval
            dAv = Vector{typeof(zero(eltype(db)) * zero(eltype(dc)))}(undef, length(A.nzval))
            Threads.@threads for j = 1:n
                 @inbounds dAv[pp[j]:pp[j+1]-1] = -db[ii[pp[j]:pp[j+1]-1]] .* c[j]
             end

            dA = SparseMatrixCSC(m,n,pp,ii,dAv)
        end
        return (NO_FIELDS, dA, db)
    end
    return c, mybackslash_pullback
end

# assume every column has at least one element
function getRepeatIdx(I, J)
    V = 1:length(I)
    A = sparse(I, J, V)
    vidx = Vector{Int}(undef, length(V))
    for i in 1:length(I)
        for j in A.colptr[J[i]]:A.colptr[J[i]+1]-1
            if I[i] == A.rowval[j]
                vidx[i] = j
                break
            end
        end
    end
    return vidx
end

vidx = getRepeatIdx(iK, jK)

function ChainRulesCore.rrule(
    ::typeof(SparseArrays.sparse),
    I::Vector, J::Vector, Av::Vector
)
    A = sparse(I, J, Av)
    function SparseMatrixCSC_pullback(dA::SparseMatrixCSC)
        return (NO_FIELDS, DoesNotExist(), DoesNotExist(), @thunk(dA.nzval[vidx]))
    end

    return A, SparseMatrixCSC_pullback
end

function top2D(x)::Float64
      sK = reshape(Kₑ[:]*(x[:]'.^penalty .+ Emin), 8*ex*8*ey)
      sK1 = sK[idx1]

      K = sparse(iK,jK,sK1)

      U = mybackslash(K, F)

      loss = abs(U[fPoint])
end

k1 = Int(ey/4)
k2 = Int(ex/4)
ch1 = Int(k1*k2*8)

# (Hin - 1)*S - 2P + K - 1 + 1
model = Chain(Dense(1, ch1),
              x -> reshape(x, k1, k2, 8, :),
              ConvTranspose((4, 4), 8 => 4,  stride = 2, pad = 1),
              ConvTranspose((4, 4), 4 => 1,  stride = 2, pad = 1),
              Conv((filter1, filter1), 1 => 1, stride = 1,tanh, pad = SamePad()),
              ) |> f64

for i in 1:filter1^2
    params(model)[7][i] = 1/filter1^2
end
θ = params(model[1],model[3:4])

@views function scaleWeight(x::AbstractArray)
        x = (x[:] .+ 1)./2
    end

@views function weightLoss(x::AbstractArray)
  x_frac = sum(x[:])/etot
  wl = abs(x_frac - WEIGHT_FRAC)
end


x0 =  repeat([1.0], ey, ex)

Lref = top2D(x0)

println("")
println("worst performance: ", Lref);

function check_design(model, iter, resultFile)
    x = model([WEIGHT_FRAC])
    x = scaleWeight(x)
    wl = weightLoss(x)
    x = reshape(x, ey,ex)
    save(File(format"PNG", "results//$resultFile//x_$iter.png"), colorview(Gray, 1 .- x))
    return x, wl
end

function my_custom_train!(ps, data, opt)
  loss_w = Vector{Float64}()
  loss_c = Vector{Float64}()
  w_count = 0
  local training_loss
  local objFlag
  d, state = iterate(data)
  while length(loss_c) < n_train
     gs = gradient(ps) do
             x = model(d)
             x = scaleWeight(x)
             wl = weightLoss(x)
             if wl > 0.005
               objFlag = false
               training_loss = wl
             else
               objFlag = true
               training_loss = top2D(x)/Lref
             end
      # Insert what ever code you want here that needs Training loss, e.g. logging
    end
    # insert what ever code you want here that needs gradient
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge
    if !objFlag
        push!(loss_w, training_loss)
        w_count += 1
    else
        push!(loss_c, training_loss)
        println("iter: ", length(loss_c), "  ", d, "  training loss: ", training_loss)
        open("results//$resultFile//summary.txt", "a+") do io
                println(io, "iter: ", length(loss_c), "  ", d, "  training loss: ", training_loss)
               end;
        d, state = iterate(data)
        w_count = 0
    end
    update!(opt, ps, gs)
    # Here you might like to check validation set accuracy, and break out to do early stopping
    if length(loss_c) % save_every == 1
      check_design(model, length(loss_c), resultFile)
      #println(@sprintf("    print every: %.1f ", count))
    end
  end
end

opt = ADAM(0.01)

if randomInput == 0
    modelIn() = reshape([WEIGHT_FRAC], 1,1)
else
    modelIn() = reshape(rand(1), 1,1)
end

dataset = (modelIn() for i = 1:n_train)

x_opt1, wl1 = check_design(model, "before", resultFile)

open("results//$resultFile//summary.txt", "w") do io
           #write(io, "Hello world!")
           println(io, "penalty: ", penalty)
           println(io, "last convolution filter size: ", filter1)
           println(io, "random input?: ", randomInput)
           println(io, "weight fraction: ", WEIGHT_FRAC)
       end;

@time my_custom_train!(θ, dataset, opt)

x_opt2, wl2 = check_design(model, "after", resultFile)

println("weight loss 1: ", wl1)
println("current performance 1: ", top2D(x_opt1))
println("weight loss 2: ", wl2)
println("current performance 2: ", top2D(x_opt2))

open("results//$resultFile//summary.txt", "a+") do io
           #write(io, "Hello world!")
           println(io, "worst performance: ", Lref);
           println(io, "weight loss 1: ", wl1)
           println(io, "current performance 1: ", top2D(x_opt1))
           println(io, "weight loss 2: ", wl2)
           println(io, "current performance 2: ", top2D(x_opt2))
       end;
