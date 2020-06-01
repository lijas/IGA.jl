using JuAFEM
using IGA

function generate_square_with_hole(nel::NTuple{2,Int}, L::T=4.0, R::T = 1.0) where T

    grid = generate_grid(Quadrilateral, nel)

    nnodesx = nel[1] + 1
    nnodesy = nel[2] + 1

    angle_range = range(0.0, stop=pi/2, length = nnodesy*2 - 1)

    nodes = Node{2,T}[]
    for i in 1:nnodesy
        a = angle_range[i]

        k = tan(a)
        y = k*L

        x_range = range(-sqrt(L^2 + y^2), stop = -R, length=nnodesx)

        for x in x_range
            v = Vec{2,T}((x*cos(a), -sin(a)*x))
            push!(nodes, Node(v))
        end
    end

    #part2
    offset = (nnodesx*nnodesy) - nnodesx
    cells = copy(grid.cells)
    for cell in cells
        new_node_ids = cell.nodes .+ offset
        push!(grid.cells, Quadrilateral(new_node_ids))
    end

    
    for i in (nnodesy+1):(2*nnodesy-1)
        a = pi/2 - angle_range[i]
        k = tan(a)
        y = k*L
        x_range = range(-sqrt(L^2 + y^2), stop = -R, length=nnodesx)

        for x in x_range
            v = Vec{2,T}((sin(a)*x, -x*cos(a)))
            push!(nodes, Node(v))
        end
    end


    JuAFEM.copy!!(grid.nodes, nodes)
    return grid
end

function main()

    grid = generate_square_with_hole((15,15))
    vtk_grid("test_grid", grid) do vtk
        
    end
end