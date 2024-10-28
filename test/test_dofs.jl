

@testset "Multi-fields for iga" begin
    grid = generate_grid(BezierCell{RefQuadrilateral,2}, (2,2))

    ip = IGAInterpolation{RefQuadrilateral,2}()
    dh = DofHandler(grid)
    add!(dh, :u, ip^2)
    add!(dh, :p, ip)
    close!(dh)

    urange = dof_range(dh, :u)
    prange = dof_range(dh, :p)
    a = zeros(Float64, ndofs(dh))

    dofs = zeros(Int, ndofs_per_cell(dh))
    for cellid in 1:getncells(grid)
        celldofs!(dofs, dh, cellid)
        a[dofs[urange]] .= 1.0
        a[dofs[prange]] .= 2.0
    end

    VTKIGAFile("test_export", grid) do vtk 
        write_solution(vtk, dh, a)
    end

    #TODO: Add tests for VTK?
end