using IGA
using Ferrite

function blabla()

    L = 10.0
    h = 0.1
    nelx = 100
    nely = 1
    order = 3
    dim = 2

    mesh = IGA.generate_nurbs_patch((nelx, nely), (order,order), (L,h))
    grid = IGA.BezierGrid(mesh)
    addnodeset!(grid, "left", (x) -> x[1]≈0.0)
    addnodeset!(grid, "top", (x) -> x[2]≈h)
    addfaceset!(grid, "top", (x) -> x[2]≈h)

    file = open("beam_order$(order)_nelx$(nelx)_nely$(nely)", "w")

    println(file, "<Nodes>")
    for (nodeid, node) in enumerate(grid.nodes)

        s = "$(nodeid-1)"
        for i in 1:dim
            s *= "  $(node.x[i])"
        end
        s*=";"
        println(file, s)
    end
    println(file, "</Nodes>")

    println(file, "<Elements>")
    for (cellid, cell) in enumerate(grid.cells)
        s = "$(cellid-1)"
        for nodeid in cell.nodes
            s *= "  $(nodeid-1)"
        end
        s *= ";"
        println(file, s)
    end
    println(file, "</Elements>")

    println(file, " ")
    println(file, " ")

    println(file, "<NodeDatabase name=\"ControlPoint\">")
    println(file, "  <Column name = \"weight\" type = \"float\">")

    for (nodeid, node) in enumerate(grid.nodes)
        println(file, "    $(nodeid-1) 1.0;")
    end
    println(file, "  </Column>")
    println(file, "</NodeDatabase>")

    println(file, " ")
    println(file, " ")

    println(file, "<ElementDatabase name=\"C\">")
    #=println(file, "  <Column name = \"type\" type = \"int\">")
    for cellid in 1:Ferrite.getncells(grid)
        println(file,"    $(cellid-1) 62;")
    end
    println(file, "  </Column>")=#

    println(file, " ")

    I = Dict{Int, Vector{Int}}()
    J = Dict{Int, Vector{Int}}()
    V = Dict{Int, Vector{Float64}}()

    for cellid in 1:Ferrite.getncells(grid)
        beo = grid.beo[cellid]
        I[cellid] = Int[]
        J[cellid] = Int[]
        V[cellid] = Float64[]
        for r in 1:length(beo)
            for (i, nz_ind) in enumerate(beo[r].nzind)                
                push!(I[cellid], r-1)
                push!(J[cellid], nz_ind-1)
                push!(V[cellid], beo[r].nzval[i])
            end
        end
    end

    println(file,"  <Column name = \"i\" type = \"int\">")
    for cellid in 1:Ferrite.getncells(grid)
        s = "    $(cellid-1)  "
        for indx in I[cellid]
            s*= string(indx)*" "
        end
        s*=";"
        println(file,s)
    end
    println(file, "  </Column>")

    println(file, " ")

    println(file,"  <Column name = \"j\" type = \"int\">")
    for cellid in 1:Ferrite.getncells(grid)
        s = "    $(cellid-1)  "
        for indx in J[cellid]
            s*= string(indx)*" "
        end
        s*=";"
        println(file, s)
    end
    println(file, "  </Column>")

    println(file, " ")

    println(file,"  <Column name = \"v\" type = \"float\">")
    for cellid in 1:Ferrite.getncells(grid)
        s = "    $(cellid-1)  "
        for indx in V[cellid]
            s*= string(indx)*" "
        end
        s*=";"
        println(file, s)
    end
    println(file, "  </Column>")
    println(file, "</ElementDatabase>")

    println(file, " ")

    for key in keys(getnodesets(grid))
        println(file, "<NodeGroup name=\"$(string(key))\">")
        s = "    {"
        for nodeid in getnodeset(grid, key)
            s *= " $(nodeid-1) "
        end
        s*="}"
        println(file, s)
        println(file, "</NodeGroup>")

        println(file, " ")
    end

    for key in keys(getfacesets(grid))
        println(file, "<FaceGroup name=\"$(string(key))\">")
        s = "    {"
        for face in getfaceset(grid, key)
            s *= " ($(face[1]-1), $(face[2]-1)) "
        end
        s*="}"
        println(file, s)
        println(file, "</NodeGroup>")

        println(file, " ")
    end

    println(file, " ") 

    println(file, "<NodeConstraints>")
    println(file, "   dx[\"left\"] = 0.0")
    println(file, "   dy[\"left\"] = 0.0")
    println(file, "</NodeConstraints>")

    println(file, " ")
    println(file, " ")

    println(file, "<NodeTable name = \"load1\">")
    println(file, "  <Section columns = \"dy\">")
    println(file, "  top  10.0")
    println(file, "  </Section>")
    println(file, "<NodeTable>")
    
    close(file)

end