

@testset "iga cell cache" begin

    grid = generate_grid(BezierCell{RefQuadrilateral,2}, (3,3))

    ip = IGAInterpolation{RefQuadrilateral,2}()

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    cc = IGACellCache(dh)

    cell_id = 5
    reinit!(cc, cell_id)

    cell = getcells(grid, cell_id)
    nodes = collect(Ferrite.get_node_ids(cell))
    dofs = zeros(Int, Ferrite.ndofs_per_cell(dh, cell_id))
    celldofs!(dofs, dh, cell_id)
    coords = getcoordinates(grid, cell_id)

    @test Ferrite.getnodes(cc) == nodes
    @test getcoordinates(cc).xb == coords.xb
    @test getcoordinates(cc).x == coords.x
    @test getcoordinates(cc).wb == coords.wb
    @test getcoordinates(cc).w == coords.w
    @test celldofs(cc) == dofs
    @test cellid(cc) == cell_id

end