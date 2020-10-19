using JuAFEM
using IGA

struct MyLagrange{dim,shape,order} <: Interpolation{dim,shape,order} end

function _lagrange_ordering(::MyLagrange{2,RefCube,order}) where {order}
    dim = 2
    orders = ntuple(_-> order, dim)

    ci = CartesianIndices((orders.+1))
    ind = reshape(1:(order+1)^dim, (orders.+1)...)

    #Corners
    ordering = Int[]
    corner = ci[1,1]
    push!(ordering, ind[corner])

    corner = ci[end,1]
    push!(ordering, ind[corner])

    corner = ci[end,end]
    push!(ordering, ind[corner])

    corner = ci[1,end]
    push!(ordering, ind[corner])

    #edges
    edge = ci[2:end-1,1]
    append!(ordering, ind[edge])
    
    edge = reverse(ci[end,2:end-1])
    append!(ordering, ind[edge])

    edge = reverse(ci[2:end-1,end])
    append!(ordering, ind[edge])

    edge = reverse(ci[1,2:end-1])
    append!(ordering, ind[edge])

    #inner dofs, ordering??
    rest = ci[2:end-1,2:end-1]
    append!(ordering, ind[rest])
    return ordering
end

function test_export()

    dim = 2
    order = 3

    Lx = 1.0
    Ly = 1.0*2

    nurbsmesh = IGA.generate_nurbs_patch((1 + order, 1 + order),(order,order),(Lx,Ly))
    grid = IGA.convert_to_grid_representation(nurbsmesh)

    for node in grid.nodes

        x,y = node.x
        println("$x $y 0.0")

    end

    order = _lagrange_ordering(MyLagrange{2,RefCube,order}())

    for cell in grid.cells
        for nodeid in cell.nodes
            nid = order[nodeid]
            print("$(nid-1)")
            print(" ")
        end
    end

end